from __future__ import print_function
import os
import argparse
import shutil
import yaml
from mmcv import Config
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import mmcv
import time

from models.backbones.resnet import *
from methods.train_modes import trades_train, pgd_adv_train, plain_train, \
    mixup_vertex_train, inner_loss_train, add_reg_train
from datasets.builder import build_datasets
from datasets.loader.build_loader import build_dataloader
from utils.env import get_root_logger, set_random_seed, set_default_configs, \
    load_checkpoint, init_dist, logger_info, _CustomDataParallel
from models.Networks import Networks
from attacks.pgd_attack import eval_adv_test_whitebox_pgd
from attacks.other_attacks import eval_adv_test_whitebox_full
from attacks.auto_attack import eval_auto_attack


parser = argparse.ArgumentParser(description='PyTorch CIFAR-LT Adversarial Training')
parser.add_argument('config',
                    default='./configs/cifar10_plain.yaml',
                    help='path to config file')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--rename', '-r', action='store_true', default=False,
                    help='whether allow renaming the checkpoints parameter to match')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                    default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


set_random_seed(args.seed)

# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
    device = torch.device("cuda")
else:
    distributed = True
    init_dist(args.launcher)
    local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
print("Using", torch.cuda.device_count(), "GPUs.")

# set configs 
with open(args.config) as cf:
    cfgs = Config(yaml.safe_load(cf))
if not os.path.exists(cfgs.model_dir):
    os.makedirs(cfgs.model_dir)
shutil.copyfile(args.config, os.path.join(cfgs.model_dir, args.config.split('/')[-1]))
set_default_configs(cfgs)

# setup logger
logger = get_root_logger(cfgs.log_level, cfgs.model_dir)
logger_info(logger, distributed, "Loading config file from {}".format(args.config))
logger_info(logger, distributed, "Models saved at {}".format(cfgs.model_dir))
writter = SummaryWriter(cfgs.model_dir + '/tensorboard')

# setup data loader
logger_info(logger, distributed, "Building datasets {} with imbalance ratio {} / existing ratio".format(
    cfgs.dataset, cfgs.imbalance_ratio, cfgs.existing_ratio))
num_classes=cfgs.num_classes
trainset, samples_per_cls = build_datasets(name=cfgs.dataset, mode='train',
                                           num_classes=num_classes,
                                           imbalance_ratio=cfgs.imbalance_ratio,
                                           existing_ratio=cfgs.existing_ratio,
                                           root='../data')
testset, _ = build_datasets(name=cfgs.dataset, mode='test',
                            num_classes=num_classes, root='../data')

train_loader = build_dataloader(trainset, imgs_per_gpu=cfgs.batch_size, dist=distributed, sampler=cfgs.sampler, shuffle=True)
test_loader = build_dataloader(testset, imgs_per_gpu=cfgs.test_batch_size, dist=distributed, shuffle=False)
plain_train_loader = build_dataloader(trainset, imgs_per_gpu=cfgs.batch_size, dist=distributed, shuffle=False)


def train(cfgs, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    model.epoch = epoch
    start = time.time()
    samples_total = len(train_loader) * cfgs.batch_size
    # loss_func = Losses(samples_per_cls, num_classes, cfgs.loss_opt)
    for batch_idx, (data, target) in enumerate(train_loader):
        if not cfgs.cpu_data:
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss_params = dict(
            model=model, x_natural=data, y=target, optimizer=optimizer,
            step_size=cfgs.train_step_size, epsilon=cfgs.train_epsilon,
            perturb_steps=cfgs.train_num_steps, targeted=cfgs.targeted,
            target_opt=cfgs.target_opt, samples_per_cls=samples_per_cls,
            deffer_opt=cfgs.deffer_opt
        )

        if 'plain' == cfgs.train_mode:
            loss = plain_train(**loss_params)
        elif 'pgd_at' == cfgs.train_mode:
            other_params = cfgs.other_params
            loss = pgd_adv_train(**loss_params, **other_params)
        elif 'trades' == cfgs.train_mode:
            loss = trades_train(beta=cfgs.beta, **loss_params)
        elif 'AVmix' == cfgs.train_mode:
            loss = mixup_vertex_train(**loss_params)
        elif 'inner_loss' == cfgs.train_mode:
            loss = inner_loss_train(**loss_params)
        elif 'add_reg' in cfgs.train_mode:
            extra_params = cfgs.train_mode['add_reg']
            loss = add_reg_train(**loss_params, **extra_params)
        else:
            raise NameError
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, distributed, 'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}  time:{:.3f}'.format(
                epoch, batch_idx * len(data), samples_total,
                100. * batch_idx / len(train_loader), loss.item(),
                time.time()-start))
    if 'our' in cfgs.train_mode:
        if distributed:
            model.module.classifier.on_epoch()
        else:
            if hasattr(model.classifier, 'on_epoch'):
                model.classifier.on_epoch()
                print('Operating on epoch!')

def eval_train(cfgs, model, device, train_loader, logger):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    logger_info(logger, distributed, 'Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader) / cfgs.batch_size))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def eval_test(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    logger_info(logger, distributed, 'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def feature_ext(cfgs, model, device, data_loader, logger, feature_path):
    """ extract and save features """
    model.eval()
    samples_total = len(data_loader) * cfgs.batch_size
    print('sample_total {}, num_iterations {}'.format(samples_total, len(data_loader)))
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        feature = model.backbone(data)
        features.append(feature.cpu().detach().numpy())
        labels.extend(target.cpu().detach().numpy().tolist())
    features = np.vstack(features)
    print('feature shape: ', features.shape, 'label shape: ', len(labels))
    mmcv.dump(dict(features=features, labels=labels), feature_path)
    logger_info(logger, distributed, 'Feature extracted with {} samples and saved at {}'.format(
            samples_total, feature_path))

def adjust_learning_rate(optimizer, epoch, lr_decline=(60, 75, 90), warmup_epochs=0):
    """ decrease the learning rate """
    decay = 1.
    for epoch_d in lr_decline:
        if epoch >= epoch_d:
            decay *= 0.1
    if decay < 1.:
        logger_info(logger, distributed, "learning rate decay by {:.3f}".format(decay))
    if epoch < warmup_epochs:
        decay = (epoch + 1) / warmup_epochs

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def assign_learning_rate(model, lr_dict):
    """ assign learning rate for different parts
    lr_dict: {module key word: lr}
    """
    params = []
    logger_info(logger, distributed, "set specific learning rate: ")
    text = ""
    for k, v in lr_dict.items():
        text += "{} : {}\t".format(k, v)
    logger_info(logger, distributed, text)
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        part = name.split('.')[0]
        for key, lr in lr_dict.items():
            if key in name:
                param_group['lr'] = lr
                print(name, end=' ')
                break
        if 'lr' not in param_group:
            param_group['lr'] = lr_dict['else']
        params.append(param_group)
    print("done.")
    return params

def main():
    # init model
    model = Networks(cfgs, num_classes=num_classes, samples_per_cls=samples_per_cls).cuda()

    if distributed:
        raise NotImplementedError
        # # Currently distributed training is not well supported
        # # because the performance would drop for some unknown reason.
        # sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     sync_bn_model, device_ids=[local_rank], output_device=local_rank)
        # print("local rank: {}".format(local_rank))
    else:
        model = _CustomDataParallel(model)

    logger_info(logger, distributed, 'num_samples {}, num_iterations {}'.format(
        len(trainset), len(train_loader)))

    # load checkpoint
    load_path = None
    if cfgs.load_model is not None:
        load_path = cfgs.load_model
        print('here')
    elif cfgs.resume_epoch > 0:
        load_path = os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(cfgs.resume_epoch))
    if load_path is not None:
        assert os.path.exists(load_path), load_path
        load_checkpoint(model, logger, load_path, load_modules=cfgs.load_modules, rename=args.rename)
        if getattr(model, 'load_model', None):
            model.load_model(cfgs.model_dir)

    # extract feature and calculate centroids if needed
    feature_path = cfgs.model_dir + '/feature.pkl'
    if cfgs.feature_ext:
        feature_ext(cfgs, model, device, plain_train_loader, logger, feature_path)
    if cfgs.centroids:
        model.classifier.centroids_cal(feature_path)

    # assign learning rate
    if cfgs.lr_dict is not None:
        params = assign_learning_rate(model, cfgs.lr_dict)
        optimizer = optim.SGD(params, lr=cfgs.lr, momentum=cfgs.momentum, weight_decay=cfgs.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfgs.lr, momentum=cfgs.momentum, weight_decay=cfgs.weight_decay)

    start_epoch = cfgs.resume_epoch + 1
    for epoch in range(start_epoch, cfgs.epochs + 1):

        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch, cfgs.lr_decline, cfgs.warmup_epochs)

        # adversarial training
        train(cfgs, model, device, train_loader, optimizer, epoch, logger)

        # evaluation
        eval_batches = 20
        # since we choose the model from the last epoch in all the experiments without the strategy of early-stop
        # we only evaluate part of the test set during training (could also raise the number to eval the whole.).
        if epoch % cfgs.eval_freq == 0 or epoch == 76:
            print('================================================================')
            train_loss, train_accuracy = eval_train(cfgs, model, device, train_loader, logger)
            test_loss, test_accuracy = eval_test(model, device, test_loader, logger)
            writter.add_scalars('acc', {'train': train_accuracy, 'test': test_accuracy}, epoch)
            writter.add_scalars('loss',  {'train': train_loss, 'test': test_loss}, epoch)
            eval_adv_test_whitebox_pgd(model, device, cfgs, logger, test_loader, num_classes, mode='train', eval_batches=eval_batches)
            eval_adv_test_whitebox_full(model, cfgs, device, logger, test_loader, 'CW', eval_batches=eval_batches)
            print('================================================================')

        # save checkpoint
        if epoch % cfgs.save_freq == 0 or epoch == 76: #and epoch >= cfgs.begin_save:
            if getattr(model.classifier, 'save_model', None):
                model.classifier.save_model(cfgs.model_dir)
            # save extra parameters
            if not distributed:
                torch.save(model.state_dict(),
                       os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(epoch)))
            else:
                if local_rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfgs.model_dir, 'epoch{}.pt'.format(epoch)))

    eval_adv_test_whitebox_pgd(model, device, cfgs, logger, test_loader, num_classes)
    eval_adv_test_whitebox_full(model, cfgs, device, logger, test_loader, 'CW')
    eval_adv_test_whitebox_full(model, cfgs, device, logger, test_loader, 'MIM')
    eval_auto_attack(model, device, cfgs, logger, test_loader)
    logger_info(logger, distributed, '[Remarks] {} | End of training, saved at {}'.format(cfgs.remark, cfgs.model_dir))
writter.close()


if __name__ == '__main__':
    main()
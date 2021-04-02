from __future__ import print_function
import argparse
import yaml
import os
import shutil
import torch
from mmcv import Config, mkdir_or_exist

from models.Networks import *
from utils.env import get_root_logger, set_default_configs, load_checkpoint, init_dist
from datasets.builder import build_datasets
from attacks.pgd_attack import eval_adv_test_whitebox_pgd, eval_clean_only
from attacks.auto_attack import eval_auto_attack
from datasets.loader.build_loader import build_dataloader
from attacks.other_attacks import eval_adv_test_whitebox_full

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('config',
                    default='./configs/cifar10_plain.yaml',
                    help='path to config file')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use')
parser.add_argument('--rename', '-r', action='store_true', default=False,
                    help='whether allow renaing the checkpoints parameter to match')
parser.add_argument('--from_file', '-f', action='store_true', default=False,
                    help='analysis data from file')
parser.add_argument('--eval_train_data', action='store_true', default=False,
                    help='whether eval train data')
parser.add_argument('--save_features', '-s', action='store_true', default=True,
                    help='whether save features')
parser.add_argument('--individual', action='store_true', default=False,
                    help='whether to perform individual aa')
parser.add_argument('--attacker', '-a', default='ALL', # ['ALL', 'PGD']
                    help='which attack to perform')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


# settings
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set configs
with open(args.config) as cf:
    cfgs = Config(yaml.safe_load(cf))
mkdir_or_exist(cfgs.model_dir)
shutil.copyfile(args.config, os.path.join(cfgs.model_dir, "config_test.yaml"))
set_default_configs(cfgs)

# setup logger
logger = get_root_logger(cfgs.log_level, cfgs.model_dir)
logger.info("Loading config file from {}".format(args.config))
logger.info("Work_dir: {}".format(cfgs.model_dir))

# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launche)


def main():
    # set up data loader
    logger.info("Building test datasets {}".format(cfgs.dataset))
    num_classes=cfgs.num_classes
    trainset, samples_per_cls = build_datasets(name=cfgs.dataset, mode='train',
                                               num_classes=num_classes,
                                               imbalance_ratio=cfgs.imbalance_ratio,
                                               root='../data')
    train_loader = build_dataloader(trainset, imgs_per_gpu=cfgs.test_batch_size, dist=False, shuffle=False)
    testset, _ = build_datasets(name=cfgs.dataset, mode='test',
                                num_classes=num_classes, root='../data')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfgs.test_batch_size, shuffle=False, **kwargs)

    if args.eval_train_data: # for some statistics
        mode = 'train'
        loader = train_loader
    else:
        mode = 'test'
        loader = test_loader

    if cfgs.white_box_attack:
        # white-box attack
        logger.info('pgd white-box attack')
        logger.info('Loading from {}'.format(cfgs.model_path))
        model = Networks(cfgs, num_classes=num_classes, samples_per_cls=samples_per_cls).to(device)
        # load checkpoint
        load_checkpoint(model, logger, cfgs.model_path, rename=args.rename)
        model.eval()

        eval_clean_only(model=model, device=device, logger=logger, test_loader=loader, cfgs=cfgs)

        if args.attacker == 'PGD':
            #  PGD Attack
            eval_adv_test_whitebox_pgd(model, device, cfgs, logger, loader, num_classes,
                                       targeted=cfgs.targeted, save_features=args.save_features,
                                       mode=mode)
        elif args.attacker == 'ALL':
            # CW Attack
            eval_adv_test_whitebox_full(model, cfgs, device, logger, loader, 'CW')
            # MIM Attack
            eval_adv_test_whitebox_full(model, cfgs, device, logger, loader, 'MIM')
            # Auto Attack
            eval_auto_attack(model, device, cfgs, logger, loader, individual=args.individual)
            # PGD Attack
            # eval_adv_test_whitebox_full(model, cfgs, device, logger, loader, 'PGD', early_stop)
            # PGD Attack
            eval_adv_test_whitebox_pgd(model, device, cfgs, logger, loader, num_classes,
                                       targeted=cfgs.targeted, save_features=args.save_features,
                                       mode=mode , print_freq=100)
        else:
            raise NameError
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()

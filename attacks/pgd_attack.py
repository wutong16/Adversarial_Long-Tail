from __future__ import print_function
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import recall_score
from models.backbones.resnet import *
from utils.data_utils import save_csv, CountMeter
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import os
import mmcv


def _pgd_whitebox_basic(
        model,
        device,
        X,
        y,
        random,
        epsilon,
        num_steps,
        step_size,
        eval_steps = (5,10),
        by_class=False,
        num_classes=10,
        **kwargs
        ):
    model.eval()

    out = model(X)

    outputs = dict(CLEAN=out.clone().detach().cpu().numpy())

    # FGSM attack
    X_fgsm = Variable(X.data, requires_grad=True)
    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()
    outputs.update({'zero_grad': (100 * torch.sum(X_fgsm.grad==0) / torch.sum(torch.ones_like(X_fgsm))).detach().cpu().numpy()})
    eta = epsilon * X_fgsm.grad.data.sign()
    X_fgsm = Variable(X.data + eta, requires_grad=True)
    X_fgsm = Variable(torch.clamp(X_fgsm, 0, 1.0), requires_grad=True)
    outputs.update({'FGSM':model(X_fgsm).detach().cpu().numpy()})

    # PGD attack
    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    avg_grad_nat = 0
    avg_grad_rob = 0
    for i_step in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():

            pgd_out = model(X_pgd) # after i_step's attack

            if i_step in eval_steps:
                outputs.update({'PGD-{}'.format(i_step):pgd_out.detach().cpu().numpy()})

            loss = nn.CrossEntropyLoss()(model(X_pgd), y)

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        if i_step == 0:
            avg_grad_nat += torch.sum(torch.abs(X_pgd.grad.data)) / torch.sum(torch.ones_like(X_pgd))
        elif i_step == num_steps - 1:
            avg_grad_rob += torch.sum(torch.abs(X_pgd.grad.data)) / torch.sum(torch.ones_like(X_pgd))
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    outputs.update({'PGD-{}'.format(num_steps):model(X_pgd).detach().cpu().numpy()})

    return outputs

def _pgd_whitebox_extension(
        model,
        device,
        X,
        y,
        random,
        epsilon,
        num_steps,
        step_size,
        step_counters=None,
        targeted=False,
        fix_target=None,
        eval_steps = (5,10),
        save_features = False,
        eval_fine=False,
        by_class=False,
        num_classes=10,
        **kwargs
        ):
    model.eval()

    out = model(X)

    outputs = dict(CLEAN=out.clone().detach().cpu().numpy())

    if fix_target is not None:
        new_y = fix_target
    else:
        tmp1 = torch.argsort(out, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    # FGSM attack
    X_fgsm = Variable(X.data, requires_grad=True)
    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()
    outputs.update({'zero_grad': (100 * torch.sum(X_fgsm.grad==0) / torch.sum(torch.ones_like(X_fgsm))).detach().cpu().numpy()})
    eta = epsilon * X_fgsm.grad.data.sign()
    X_fgsm = Variable(X.data + eta, requires_grad=True)
    X_fgsm = Variable(torch.clamp(X_fgsm, 0, 1.0), requires_grad=True)
    outputs.update({'FGSM':model(X_fgsm).detach().cpu().numpy()})

    # PGD attack
    X_pgd = Variable(X.data, requires_grad=True)
    success_steps = np.zeros(X.shape[0])

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    avg_grad_nat = 0
    avg_grad_rob = 0
    for i_step in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():

            pgd_out = model(X_pgd) # after i_step's attack

            if i_step in eval_steps:
                outputs.update({'PGD-{}'.format(i_step):pgd_out.detach().cpu().numpy()})

            if step_counters is not None:
                if targeted:
                    loss = - nn.CrossEntropyLoss(reduction='none')(model(X_pgd), new_y)
                else:
                    loss = nn.CrossEntropyLoss(reduction='none')(model(X_pgd), y)
                step_counters[i_step].update (loss.clone().detach().cpu().numpy(), y.cpu().numpy())
                loss = loss.mean()

            elif eval_fine:
                if targeted:
                    logits = model(X_pgd)
                    success = (logits.data.max(1)[1] == new_y.data).cpu().numpy()
                    success_steps[success * (success_steps==0)] = i_step + 1
                    loss = - nn.CrossEntropyLoss()(logits, new_y)
                else:
                    logits = model(X_pgd)
                    error = (logits.data.max(1)[1] != y.data).cpu().numpy()
                    success_steps[error * (success_steps==0)] = i_step + 1
                    loss = nn.CrossEntropyLoss()(logits, y)

            else:
                if targeted:
                    loss = - nn.CrossEntropyLoss()(model(X_pgd), new_y)
                else:
                    loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        if i_step == 0:
            avg_grad_nat += torch.sum(torch.abs(X_pgd.grad.data)) / torch.sum(torch.ones_like(X_pgd))
        elif i_step == num_steps - 1:
            avg_grad_rob += torch.sum(torch.abs(X_pgd.grad.data)) / torch.sum(torch.ones_like(X_pgd))
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    outputs.update({'PGD-{}'.format(num_steps):model(X_pgd).detach().cpu().numpy()})

    if save_features:
        outputs.update(CLEAN_features=model.backbone(X).detach().cpu().numpy())
        outputs.update(PGD_features=model.backbone(X_pgd).detach().cpu().numpy())

    if eval_fine:
        outputs.update(success_steps=success_steps,
                       avg_grad_nat=avg_grad_nat.detach().cpu().numpy(),
                       avg_grad_rob=avg_grad_rob.detach().cpu().numpy())

    return outputs

def eval_metrics(y_out, y_true, num_classes=10, samples_per_cls=None):

    y_pred = np.argmax(y_out,1)
    y_true = np.asarray(y_true)
    precision, recall, f_score, true_sum = \
        precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    return accuracy, precision, recall, f_score, true_sum


def eval_clean_only(model, device, cfgs, logger, test_loader):

    model.eval()

    all_labels = []
    all_outputs = []

    for i, (data, target) in enumerate(test_loader):

        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        outputs = model(X).detach().clone().cpu().numpy()
        all_outputs.append(outputs)
        all_labels.extend(target.cpu().detach().numpy().tolist())

    all_outputs = np.vstack(all_outputs)
    accuracy, precision, recall, f_score, true_sum = eval_metrics(all_outputs, all_labels)

    logger.info('>>> clean accuracy: {}'.format(accuracy))
    for r in recall:
        print('{:.4f}'.format(r), end=' ')
    print()

    return

def eval_adv_test_whitebox_pgd(model, device, cfgs, logger, test_loader, num_classes=10, print_freq=60, by_class=True,
                               targeted=False, step_eval=False, save_features=False, mode='test', eval_batches=10000,
                               eval_pair_mode=None, **kwargs):
    """
    evaluate model by white-box attack
    """
    model.eval()

    total_batches = len(test_loader)
    all_labels = []
    all_outputs = dict()

    eval_steps = (5, 10, 20)

    if step_eval:
        step_counters = [CountMeter(num_classes) for _ in range(cfgs.test_num_steps)]
    else:
        step_counters= None

    total = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        total += len(data)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)

        outputs = _pgd_whitebox_basic(
            model, device, X, y, random=cfgs.random, epsilon=cfgs.test_epsilon,
            num_steps=cfgs.test_num_steps, step_size=cfgs.test_step_size, eval_steps=eval_steps,
            by_class=by_class, num_classes=num_classes, step_counters=step_counters,
            targeted=targeted, save_features=save_features
        )

        all_labels.extend(target.cpu().detach().numpy().tolist())

        for key in outputs.keys():
            if key not in all_outputs.keys():
                all_outputs.update({key: []})
                all_outputs[key].append(outputs[key])
            else:
                all_outputs[key].append(outputs[key])

        if i == 20 or (i+1) % print_freq == 0:
            logger.info("[{}/{}] batches finished".format(i+1, total_batches))
            for key, data in all_outputs.items():
                if 'zero_grad' in key:
                    # print('zero_grad_rate: {} '.format(np.mean(data)))
                    pass
                if 'features' in key or 'success' in key or 'grad' in key:
                    continue
                data = np.vstack(data)
                accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
                logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
                    key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
        if i == eval_batches:
            # sometimes we only evaluate part of the test set during training for efficiency
            logger.info(">> Only {}/{} batches are evaluated, stop here to save time.".format(i+1, total_batches))
            return

    # save metrics into .csv
    csv_name = '{}_all_results.csv'.format(cfgs.dataset)
    save_csv(csv_name, [[cfgs.model_path], [cfgs.remark]], devide=False)
    for key, data in all_outputs.items():
        save_data = [[' '], [key]]
        if 'success' in key:
            data = np.hstack(data)
        elif 'zero_grad' in key:
            data = np.mean(data)
            print(' * * * zero_grad * * * {}'.format(data))
        else:
            data = np.vstack(data)
        all_outputs[key] = data
        if 'features' in key or 'success' in key or 'grad' in key:
            continue
        accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
        # use group evaluation for CIFAR100 and ImageNet
        if recall.shape[0] >= 100:
            g_recall = np.reshape(recall, (10, -1)).sum(-1)
            g_precision = np.reshape(precision, (10, -1)).sum(-1)
            g_f_score = np.reshape(f_score, (10, -1)).sum(-1)
            save_data.extend([
                ['accuracy', accuracy],
                ['g_recall'], g_recall.tolist(),
                ['g_precision'], g_precision.tolist(),
                ['g_f_score'], g_f_score.tolist()
            ])
        # save original per-class metric
        save_data.extend([
            ['accuracy', accuracy],
            ['recall'], recall.tolist(),
            ['precision'], precision.tolist(),
            ['f_score'], f_score.tolist()
        ])
        save_csv(csv_name, save_data, devide=False)
        logger.info("Finished PGD Evaluation.")
        logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
            key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
        # for name, value in zip(['precision', 'recall', 'f_score'], [precision, recall, f_score]):
        #     print(name, end=' | ')
        #     for v in value:
        #         print("{:.2f}".format(v*100), end=' ')
        #     print()
    logger.info("[Remarks] {} | End of evaluation,model path {}".format(cfgs.remark, cfgs.model_path))

    for name, param in model.classifier.state_dict().items():
        if 'weight' in name:
            print('save classifier weight from module <{}> !'.format(name))
            all_outputs.update(CLASSIFIER_weight=model.classifier.state_dict()[name].clone().detach().cpu().numpy())

    all_outputs.update(LABLES=all_labels)
    if eval_pair_mode is not None:
        mode += '.{}'.format(eval_pair_mode)
    mmcv.dump(all_outputs, cfgs.model_path + '.{}.pkl'.format(mode))
    print('Data saved at {}'.format(cfgs.model_path + '.{}.pkl'.format(mode)))

    # save atep-wise results during PGD attacking into .csv
    if step_eval:
        for i in range(cfgs.test_num_steps):
            save_csv('./{}_all_results.csv'.format(cfgs.dataset), [[' ', ' ',' '], step_counters[i].avg_values.tolist()], msg=False)

    return

def eval_adv_test_whitebox_with_restart(model, device, cfgs, logger, test_loader, num_classes=10, print_freq=60, by_class=True,
                           targeted=False, step_eval=False, save_features=True, mode='test', num_restart=2, free_bn=False):
    """
    evaluate model by white-box attack
    """
    model.eval()

    total_batches = len(test_loader)
    all_labels = []
    all_features = dict(CLEAN_features=[], PGD_features=[])
    all_outputs = dict(CLEAN=[], FGSM=[])
    best_adv_features = dict(CLEAN_features=[], PGD_features=[])
    best_adv_outputs = dict(CLEAN=[], FGSM=[])
    best_adv_loss = dict(FGSM=[])

    if mode == 'test':
        eval_steps = (5, 10, 20, 100)
        cfgs.test_num_steps = 100
    else:
        eval_steps = (5, 10, 20)

    for i_step in eval_steps:
        all_outputs.update({'PGD-{}'.format(i_step):[]})
        best_adv_outputs.update({'PGD-{}'.format(i_step):[]})
        best_adv_loss.update({'PGD-{}'.format(i_step):[]})

    if step_eval:
        step_counters = [CountMeter(num_classes) for _ in range(cfgs.test_num_steps)]
    else:
        step_counters= None

    for i_re in range(num_restart):
        logger.info('[White box evaluation] Restart: {}'.format(i_re + 1))
        all_outputs={key: [] for key in all_outputs.keys()}
        all_labels = []
        total = 0
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            total += len(data)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)

            outputs = _pgd_whitebox_basic(
                model, device, X, y, random=cfgs.random, epsilon=cfgs.test_epsilon,
                num_steps=cfgs.test_num_steps, step_size=cfgs.test_step_size, eval_steps=eval_steps,
                by_class=by_class, num_classes=num_classes, step_counters=step_counters,
                targeted=targeted, save_features=save_features, free_bn=free_bn,
            )

            for key in all_outputs.keys():
                all_outputs[key].append(outputs[key])
            all_labels.extend(target.cpu().detach().numpy().tolist())

            if save_features:
                for key in all_features.keys():
                    all_features[key].append(outputs[key])

            if i == 20 or (i+1) % print_freq == 0:
                logger.info("[{}/{}] batches finished".format(i+1, total_batches))
                for key, data in all_outputs.items():
                    data = np.vstack(data)
                    accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
                    logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
                        key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
                if mode == 'train':
                    return

        for k, v in all_outputs.items():
            print(k)
            v = np.vstack(v)
            loss = nn.CrossEntropyLoss(reduction='none')(torch.tensor(v), torch.tensor(all_labels, dtype=torch.int64))
            if i_re == 0:
                best_adv_outputs[k] = v
                best_adv_loss[k] = loss
            else:
                # bb = best_adv_loss[k] >= loss
                bw = best_adv_loss[k] < loss
                best_adv_outputs[k][bw, :] = v[bw, :]
                best_adv_loss[k][bw] = loss[bw]

    all_outputs.update(best_adv_outputs)
    # save metrics into .csv
    csv_name = '{}_all_results.csv'.format(cfgs.dataset)
    save_csv(csv_name, [[cfgs.model_path], [cfgs.remark]], devide=False)
    for key, data in all_outputs.items():
        save_data = [[' '], [key]]
        data = np.vstack(data)
        all_outputs[key] = data
        accuracy, precision, recall, f_score, true_sum = eval_metrics(data, all_labels)
        # use group evaluation for CIFAR100 and ImageNet
        if recall.shape[0] >= 100:
            g_recall = np.reshape(recall, (10, -1)).sum(-1)
            g_precision = np.reshape(precision, (10, -1)).sum(-1)
            g_f_score = np.reshape(f_score, (10, -1)).sum(-1)
            save_data.extend([
                ['accuracy', accuracy],
                ['g_recall'], g_recall.tolist(),
                ['g_precision'], g_precision.tolist(),
                ['g_f_score'], g_f_score.tolist()
            ])
        # save original per-class metric
        save_data.extend([
            ['recall'], recall.tolist(),
            ['precision'], precision.tolist(),
            ['f_score'], f_score.tolist()
        ])
        save_csv(csv_name, save_data, devide=False)

        logger.info("metric: {} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f_score: {:.4f} | ".format(
            key, accuracy, np.mean(precision), np.mean(recall), np.mean(f_score)))
        for name, value in zip(['precision', 'recall', 'f_score'], [precision, recall, f_score]):
            print(name, end=' | ')
            for v in value:
                print("{:.2f}".format(v*100), end=' ')
            print()
    logger.info("[Remarks] {} | End of evaluation,model path {}".format(cfgs.remark, cfgs.model_path))
    if save_features:
        for key, features in all_features.items():
            all_features[key] = np.vstack(features)

    # save output and features into .pkl
    all_outputs.update(all_features)
    if hasattr(model.classifier, 'fc'):
        all_outputs.update(CLASSIFIER_weight=model.classifier.fc.weight.clone().detach().cpu().numpy())
    all_outputs.update(LABLES=all_labels)
    file_name = os.path.join(cfgs.model_dir, cfgs.model_path.split('/')[-1] + '.{}.pkl'.format(mode))
    mmcv.dump(all_outputs, file_name)
    print('Data saved at {}'.format(file_name))

    # save atep-wise results during PGD attacking into .csv
    if step_eval:
        for i in range(cfgs.test_num_steps):
            save_csv('./{}_all_results.csv'.format(cfgs.dataset), [[' ', ' ',' '], step_counters[i].avg_values.tolist()], msg=False)

    return

def pairing_attack(eval_pair_mode, num_classes):
    if 'neighbour' == eval_pair_mode:
        pairing = [(i+1)%num_classes for i in range(num_classes)]
    elif 'head_tail' == eval_pair_mode:
        pairing = [num_classes-i-1 for i in range(num_classes)]
    elif 'head_middle' == eval_pair_mode:
        middle = num_classes // 2
        pairing = [(i+middle)%num_classes for i in range(num_classes)]
    else:
        raise NameError
    return pairing

def _pgd_whitebox_pair(model,
                  logger,
                  device,
                  X,
                  y,
                  random,
                  epsilon,
                  num_steps,
                  step_size,
                  by_class=False,
                  num_classes=10,
                  step_counters=None,
                  targeted=False,
                  fix_target=None,
                  eval_steps = (5,10),
                  save_features = False,
                  **kwargs

                  ):
    model.eval()
    out = model(X)
    outputs = dict(CLEAN=out.clone().detach().cpu().numpy())

    if fix_target is not None:
        new_y = fix_target
    else:
        tmp1 = torch.argsort(out, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    # PGD attack
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    success_step = np.zeros(X.shape[0])
    for i_step in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():

            pgd_out = model(X_pgd) # after i_step's attack

            if i_step in eval_steps:
                outputs.update({'PGD-{}'.format(i_step):pgd_out.detach().cpu().numpy()})
            if step_counters is not None:
                if targeted:
                    loss = - nn.CrossEntropyLoss(reduction='none')(model(X_pgd), new_y)
                else:
                    logits = model(X_pgd)
                    success = logits
                    error = (logits.data.max(1)[1] != y.data)
                    print(error.shape)
                    exit()
                    loss = nn.CrossEntropyLoss(reduction='none')(model(X_pgd), y)
                step_counters[i_step].update (loss.clone().detach().cpu().numpy(), y.cpu().numpy())
                loss = loss.mean()
            else:
                if targeted:
                    loss = - nn.CrossEntropyLoss()(model(X_pgd), new_y)
                else:
                    loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    if save_features:
        outputs.update(CLEAN_features=model.backbone(X).detach().cpu().numpy())
        outputs.update(PGD_features=model.backbone(X_pgd).detach().cpu().numpy())

    return outputs

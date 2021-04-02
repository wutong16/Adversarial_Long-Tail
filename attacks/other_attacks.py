from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from utils.data_utils import save_csv


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  random,
                  device,
                  step_size,
                  **kwargs):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _cw_whitebox(model,
                 X,
                 y,
                 epsilon,
                 num_steps,
                 num_classes,
                 random,
                 device,
                 step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = CWLoss(100 if num_classes==100 else 10)(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err cw (white-box): ', err_pgd / len(X))
    return err, err_pgd


def _fgsm_whitebox(model,
                   X,
                   y,
                   epsilon,
                   **kwargs):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_fgsm = Variable(X.data, requires_grad=True)

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)
    err_pgd = (model(X_fgsm).data.max(1)[1] != y.data).float().sum()
    # print('err fgsm (white-box): ', err_pgd)
    return err, err_pgd

def clean(model,
                   X,
                   y,
                   epsilon):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_fgsm = Variable(X.data, requires_grad=True)

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)
    err_pgd = (model(X_fgsm).data.max(1)[1] != y.data).float().sum()
    # print('err fgsm (white-box): ', err_pgd)
    return err, err_pgd

def _mim_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device,
                  decay_factor=1.0,
                  **kwargs):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    # print('err mim (white-box): ', err_pgd / len(X))
    return err, err_pgd


def eval_adv_test_whitebox_full(model, cfgs, device, logger, test_loader, attack_method, begin_batch=0, eval_batches=10000):
    """
    evaluate model by white-box attack
    """
    logger.info("Evaluating {} Attack".format(attack_method))
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    params = dict(epsilon=cfgs.test_epsilon, num_steps=cfgs.test_num_steps,
                  step_size=cfgs.test_step_size, num_classes=cfgs.num_classes)
    total = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        if attack_method == 'PGD':
            err_natural, err_robust = _pgd_whitebox(model, X, y, random=True, device=device, **params)
        elif attack_method == 'CW':
            err_natural, err_robust = _cw_whitebox(model, X, y, random=True, device=device, **params)
        elif attack_method == 'MIM':
            err_natural, err_robust = _mim_whitebox(model, X, y, random=True, device=device, **params)
        elif attack_method == 'FGSM':
            err_natural, err_robust = _fgsm_whitebox(model, X, y, **params)
        else:
            raise NotImplementedError
        robust_err_total += err_robust
        total += len(X)
        natural_err_total += err_natural
        if i % 40 == 10:
            logger.info('{} batches evaluated, nat_acc: {:.4f}, rob_acc: {:.4f}'.format(i, 1 - natural_err_total / total, 1 - robust_err_total / total))
        if i == eval_batches:
            logger.info('{} batches evaluated, nat_acc: {:.4f}, rob_acc: {:.4f}'.format(i, 1 - natural_err_total / total, 1 - robust_err_total / total))
            break

    if eval_batches > len(test_loader):
        csv_name = '{}_all_results.csv'.format(cfgs.dataset)
        save_csv(csv_name, [[cfgs.model_path], [cfgs.remark]], devide=False)
        save_data = [[' '], [attack_method], ['accuracy', 1 - robust_err_total / total]]
        save_csv(csv_name, save_data, devide=False)

    logger.info('CLEAN : {:.4f}'.format(1 - natural_err_total / total))
    logger.info('Robust : {:.4f}'.format(1 - robust_err_total / total))
    return 1 - robust_err_total / total


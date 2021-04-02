import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

def _add_margins(epoch, input, target, samples_per_cls, opt='LDAM', epoch_devides=(0, 40), DM_maxs=(0, 0.5), **kwargs):
    if opt == 'LDAM':
        # LDAM margin
        max_m = 0
        for epoch_devide, max_margin in zip(epoch_devides, DM_maxs):
            if epoch >= epoch_devide:
                max_m = max_margin
            else:
                break
        m_list = 1.0 / np.sqrt(np.sqrt(samples_per_cls))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)

        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1)) # (batch_szie, 1)
        x_m_s = input - batch_m
        output = torch.where(index, x_m_s, input)
        # the ground_truth label won't change, other labels + margin
        # output = torch.where(index, input, x_m_b)

    else:
        raise NotImplementedError

    return output

def _add_DRW(epoch, samples_per_cls, epoch_devides=(0, 40), DRW_betas=(0, 0.9999), **kwargs):

    # idx = epoch // 60
    # betas = [0, 0.9999]
    beta = 0
    for epoch_devide, e_beta in zip(epoch_devides, DRW_betas):
        if epoch >= epoch_devide:
            beta = e_beta
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(samples_per_cls)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return per_cls_weights

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def _get_alpha(batch_size=128, opt='uniform', epoch_rate=1, p_nat=None, p_adv=None):
    if opt == 'uniform':
        alpha = torch.rand(batch_size,1,1,1).cuda()
    elif opt == 'p_uniform':
        alpha = ( 1 + p_nat) / 2 * torch.rand(batch_size).cuda()
        alpha = alpha.view(-1,1,1,1)

    elif opt == 'bias_uniform':
        alpha = torch.rand(batch_size,1,1,1).cuda()/2 + 1/4
    elif opt == 'adv_gaussian':
        alpha = 0.3 * torch.rand(batch_size,1,1,1).cuda() + 1
    elif opt == 'epoch_rate':
        alpha = (1 - epoch_rate) * (torch.rand(batch_size,1,1,1).cuda() - 0.5) + 0.5
    elif opt == 'double_uniform':
        alpha = torch.rand(batch_size).cuda() * p_nat * 0.5 + \
                torch.rand(batch_size).cuda() * p_adv * 1
        alpha = alpha.view(-1,1,1,1)
    else:
        raise NameError
    return alpha

def _get_soft_label(alpha, num_classes, labels, samples_per_cls=None, alpha_0=0.5, opt='linear', T=1):
    """ get soft label based on alpha """
    ones = torch.ones_like(alpha)

    if opt == 'linear':
        pos_value = 1 - alpha + alpha * 1 / num_classes
    elif opt == 'piecewise_linear':
        high = (num_classes - 1) / num_classes
        low = 1 / num_classes
        near = ones * high
        far = (alpha - alpha_0) / (1 - alpha_0) * (low - high) + high
        pos_value = torch.where(alpha < alpha_0, near, far)
    elif opt == 'exp_convex':
        pos_value = torch.exp(- alpha / T)
    elif opt == 'exp_concave':
        pos_value = 1 + torch.exp(-ones/T) - torch.exp((alpha-1) / T)
    elif opt == 'high_tail':
        class_freq = samples_per_cls / torch.sum(samples_per_cls)
        pos_value = 1 - alpha + alpha * class_freq.cuda()
    elif opt == 'low_tail':
        samples_per_cls = torch.tensor(samples_per_cls).float().cuda()
        class_freq = samples_per_cls / torch.sum(samples_per_cls)
        pos_value = 1 - alpha + alpha * class_freq
    elif opt == 'sigmoid':
        raise NotImplementedError
    else:
        raise NameError

    return pos_value

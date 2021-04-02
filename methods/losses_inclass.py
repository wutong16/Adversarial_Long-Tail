import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


# Loss classes
class Losses(nn.Module):

    def __init__(self, samples_per_cls, num_classes, loss_opt):
        super(Losses, self).__init__()
        if loss_opt is None:
            loss_forward = F.cross_entropy
        else:
            if "focal" in loss_opt:
                loss_forward = FocalLoss(**loss_opt["focal"])
            elif "CB" in loss_opt:
                loss_forward = CBLoss(samples_per_cls, num_classes, **loss_opt["CB"])
            elif "LDAM" in loss_opt:
                loss_forward = LDAMLoss(samples_per_cls, **loss_opt["LDAM"])
            elif "UniMargin" in loss_opt:
                loss_forward = UniMarginLoss(num_classes, **loss_opt["UniMargin"])
            elif "BCE" in loss_opt:
                loss_forward = BCELoss(num_classes, **loss_opt["BCE"])
            elif "CDT" in loss_opt:
                loss_forward = CDTLoss(samples_per_cls, **loss_opt["CDT"])
            elif "LogitAdjust" in loss_opt:
                loss_forward = LogitAdjustLoss(samples_per_cls, **loss_opt["LogitAdjust"])
            elif "MultiMargin" in loss_opt:
                loss_forward = MultiMarginLoss(samples_per_cls, **loss_opt["MultiMargin"])
            # elif "soft" in loss_opt:
            #     loss_forward = SoftLabelLoss(num_classes, **loss_opt["soft"])
            # elif "EQL" in loss_opt:
            #     loss_forward = EQL_MarginLoss(samples_per_cls, **loss_opt["EQL"])
            else:
                raise NameError
        self.loss_forward = loss_forward

    def forward(self, input, target, weight=None):
        return self.loss_forward(input, target, weight)


class FocalLoss(nn.Module):
    """ focal loss """

    def __init__(self, gamma=2.0, use_sigmoid=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        print(">> focal loss built, with gamma={}, use_sigmoid={}".format(gamma,  use_sigmoid))

    def forward(self, input, target, weight=None):
        if self.use_sigmoid:
            # original focal loss
            if len(target.shape) == 1:
                labels_onehot = F.one_hot(target, input.shape[1]).float()
            else:
                labels_onehot = target
            pt = torch.sigmoid(input * (2 * labels_onehot - 1))
            focal_loss = -((1 - pt) ** self.gamma) * torch.log(pt)
            if weight is not None:
                focal_loss *= weight
            loss = focal_loss.sum() / labels_onehot.sum()
            return loss
        else:
            # cross-entropy loss with focal-like weighting
            input_values = F.cross_entropy(input, target, reduction='none', weight=weight)
            p = torch.exp(-input_values)
            loss = (1 - p) ** self.gamma * input_values
            return loss.mean() * 0.5

class CBLoss(nn.Module):
    """ class aware weight, based on Class-Balanced (CB) Loss """

    def __init__(self, samples_per_cls, num_classes=10, beta=0.9999,
                 loss_type="focal", gamma=2.0, alpha=1., use_sigmoid=True):
        super(CBLoss, self).__init__()

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weights = torch.tensor(weights).float().unsqueeze(0).cuda()

        self.alpha = alpha
        self.num_classes = num_classes
        self.loss_type = loss_type

        if loss_type == "focal":
            self.cb_loss = FocalLoss(gamma=gamma, use_sigmoid=use_sigmoid)
        elif loss_type == "sigmoid":
            self.cb_loss = F.binary_cross_entropy_with_logits
        elif loss_type == "softmax":
            self.cb_loss = F.binary_cross_entropy
        else:
            raise NameError
        print(">> class aware weight, beta={}, loss_type={}, "
              "gamma={}, alpha={}, use_sigmoid={}".format(
            beta, loss_type, gamma, alpha, use_sigmoid))

    def forward(self, input, target, weight=None):
        labels_one_hot = F.one_hot(target, self.num_classes).float()
        weights = self.weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1, keepdim=True)
        cb_weights = weights.repeat(1, self.num_classes) # (batch_size, num_classes)
        if self.loss_type == "softmax":
            input = input.softmax(dim = 1)
        if weight is not None:
            weight = weight * cb_weights
        else:
            weight = cb_weights
        return self.cb_loss(input, labels_one_hot, weight) * self.alpha

class BCELoss(nn.Module):
    """ binary cross entropy loss """

    def __init__(self, num_classes=10, T=1):
        super(BCELoss, self).__init__()
        self.num_classes = num_classes
        self.T = T
        self.bce_loss = F.binary_cross_entropy_with_logits
        print(">> binary cross entropy loss built.")

    def forward(self, input, target, weight=None):
        labels_one_hot = F.one_hot(target, self.num_classes).float()
        return self.T * self.bce_loss(input/self.T, labels_one_hot, weight)

class LDAMLoss(nn.Module):
    """ class aware margin, based on Label-Distribution-Aware Margin (LDAM) Loss """

    def __init__(self, samples_per_cls, max_m=0.5, weight=None, s=10, inv=False, g=0.25):
        super(LDAMLoss, self).__init__()
        self.s = s
        self.weight = weight
        self.inv = inv

        m_list = 1. / np.power(samples_per_cls, g)
        if inv:
            m_list = 1 / m_list
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        # print("margin list: \n", self.m_list)

        print(">> class-aware margin with max_m={}, s={}".format(max_m, s))

    def forward(self, input, target, weight=None):
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m_s = input - batch_m
        x_m_b = input + batch_m

        if self.inv:
            output = torch.where(index, input, x_m_b)
        else:
            output = torch.where(index, x_m_s, input)

        if self.weight is not None:
            if weight is not None:
                weight *= self.weight
            else:
                weight = self.weight

        loss = F.cross_entropy(self.s * output, target, weight)

        return loss

class UniMarginLoss(nn.Module):
    """ unifrom margin, usually applied along with cosine classifier, based on CosFace """

    def __init__(self, num_classes=10, m=0.5, s=10, weight=None):
        super(UniMarginLoss, self).__init__()
        m_list =  [m] * num_classes
        self.m_list = torch.cuda.FloatTensor(m_list)
        # print("margin list: \n", self.m_list)
        self.s = s
        self.weight = weight
        print(">> uniform margin built, with m={}, s={}".format(m,s))

    def forward(self, input, target, weight=None):
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1)) # (batch_szie, 1)
        x_m_s = input - batch_m

        output = torch.where(index, x_m_s, input)

        if self.weight is not None:
            if weight is not None:
                weight *= self.weight
            else:
                weight = self.weight
        # attention: temperature applied to loss function during training, need
        # to be applied to logit output at inference time for PGD attack
        loss = F.cross_entropy(self.s*output, target, weight)

        return loss

class LogitAdjustLoss(nn.Module):
    """ class-aware bias, based on Logit Adjustment """

    def __init__(self, samples_per_cls, tau=1):
        super(LogitAdjustLoss, self).__init__()
        self.prior = torch.tensor(samples_per_cls / np.sum(samples_per_cls)).cuda()
        self.prior_bias = tau * torch.log(self.prior)
        print(">> class-aware bias built, with tau={}".format(tau))

    def forward(self, input, target, weight=None):
        input += self.prior_bias
        loss = F.cross_entropy(input, target)
        return loss

class CDTLoss(nn.Module):
    """ class-aware temperature, based on Class-Dependent Temperatures (CDT) """

    def __init__(self, samples_per_cls, tau=1):
        super(CDTLoss, self).__init__()
        self.ac = torch.tensor((np.max(samples_per_cls) / samples_per_cls) ** tau).cuda()
        # print("CDT list", self.ac)
        print(">> class-aware temperature built, with tau={}".format(tau))

    def forward(self, input, target, weight=None):
        input /= self.ac
        loss = F.cross_entropy(input, target)
        return loss

class SoftLabelLoss(nn.Module):
    """ soft label """

    def __init__(self, num_classes=10, lambda_=0.9):
        super(SoftLabelLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.neg_label = (1 - lambda_) / (num_classes-1)
        self.pos_label = lambda_

        print(">> soft label built, with lambda={}, ".format(lambda_))

    def forward(self, input, target, weight=None):

        labels_one_hot = F.one_hot(target, self.num_classes).float()
        log_softmax = F.log_softmax(input, dim=-1)
        soft_labels = labels_one_hot * self.pos_label + (1 - labels_one_hot) * self.neg_label
        loss = - soft_labels * log_softmax
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()

        return loss

class MultiMarginLoss(nn.Module):
    """ multiple margin terms, usually applied along with cosine classifier """

    def __init__(self, samples_per_cls, m=0, s=1, tau_b=0, tau_m=0):
        super(MultiMarginLoss, self).__init__()

        self.s = s
        self.use_margin = m > 0 or tau_m > 0
        m_list = torch.tensor(samples_per_cls / np.min(samples_per_cls)).cuda()
        m_list = tau_m * torch.log(m_list).float()
        self.m_list = torch.cuda.FloatTensor(m_list) / self.s + m
        # print(">> Margins: \n", self.m_list)

        if tau_b > 0:
            prior = torch.tensor(samples_per_cls / np.sum(samples_per_cls)).cuda()
            self.prior_bias = tau_b * torch.log(prior)
            # print(">> Prior bias: ", self.prior_bias)
        else:
            self.prior_bias = 0
        print(">> multiple margin terms with s={}, m={}, tau_b={}, tau_m={}".format(
            s, m, tau_b, tau_m ))

    def forward(self, input, target, weight=None):
        if self.use_margin:
            index = torch.zeros_like(input, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)
            index_float = index.type(torch.cuda.FloatTensor)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
            batch_m = batch_m.view((-1, 1))
            x_m_s = input - batch_m
            input = torch.where(index, x_m_s, input)

        input *= self.s
        input += self.prior_bias

        loss = F.cross_entropy(input, target)

        return loss

# class EQL_MarginLoss(nn.Module):
#     """ EQL loss """
#     # to do: check the implementation
#     def __init__(self, samples_per_cls, num_classes=10, thred=0.05, gamma = 0.9, s=10, inv=False, weight=None):
#         super(EQL_MarginLoss, self).__init__()
#         self.thred_mask = torch.tensor(samples_per_cls / np.sum(samples_per_cls) < thred).cuda()
#         self.thred_mask = self.thred_mask.unsqueeze(0).type(torch.uint8)
#         self.gamma = gamma
#         self.num_classes = num_classes
#         m_list =  [-1000] * num_classes
#         self.m_list = torch.cuda.FloatTensor(m_list)
#         print("margin list: \n", self.m_list)
#
#     def forward(self, input, target, weight=None):
#         self.gamma_mask = (torch.rand(input.shape) < self.gamma).type(torch.uint8).cuda()
#         index = torch.zeros_like(input, dtype=torch.uint8)
#         index.scatter_(1, target.data.view(-1, 1), 1)
#
#         index_float = index.type(torch.cuda.FloatTensor)
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
#         batch_m = batch_m.view((-1, 1))* self.gamma_mask * self.thred_mask
#         x_m_b = batch_m
#
#         output = torch.where(1- index, input, x_m_b)
#         loss = F.cross_entropy(output, target)
#         return loss


if __name__ == '__main__':
    # test
    pass


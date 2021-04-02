import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class FC_Classifier(nn.Module):
    """ plain FC classifier """

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None, focal_init=False):
        super(FC_Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

        if focal_init:
            print(">> using special bias initialization for focal loss.")
            prior = torch.tensor(samples_per_cls / np.sum(samples_per_cls)).float()
            self.fc.bias.data = -1 * torch.log((1 - prior) / prior)

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return x

class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self,  num_classes=10, in_dim=640, scale=16, bias=False):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out

class Dot_Classifier(nn.Module):
    """ dot product classifier, FC with no bias """

    def __init__(self, num_classes=10, in_dim=640):
        super(Dot_Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.fc.bias.requires_grad=False

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return x

class PostProc_Classifier(nn.Module):
    """ post processing classifier, mainly for class-aware logit bias"""

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None, tau_p=1, posthoc=False,
                 bias=False, classifier='FC', scale=10, gamma=0.03125, **kwargs):
        super(PostProc_Classifier, self).__init__()
        self.prior = torch.tensor(samples_per_cls / np.sum(samples_per_cls)).cuda()
        self.prior_bias = tau_p * torch.log(self.prior).float()
        self.posthoc = posthoc
        self.scale = scale
        self.gamma = gamma

        self.classifier = classifier
        if self.classifier == 'FC':
            self.fc = nn.Linear(in_dim, num_classes)
            self.fc.bias.requires_grad = bias
        elif self.classifier == 'Cos':
            self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
            self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        else:
            raise NotImplementedError

        print(">> post process classifier built, classifier={}, tau_p={}, scale={}".format(classifier, tau_p, scale))

        self.post_pending = True

    def forward(self, x, **kwargs):

        if self.classifier == 'FC':
            x = self.fc(x)
        elif self.classifier == 'Cos':
            # reminder: the scale and gamma for cosine classifer should be the same as train-time
            ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
            ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True) + self.gamma)
            x = torch.mm(ex, ew.t() * self.scale)

        if self.posthoc:
            x -= self.prior_bias
        # else:
        #     x += self.prior_bias # this is for logit adjustment during training, while now merged to LogitAdjustLoss
        return x

class TDESim_Classifier(nn.Module):
    """ class of feature disentangling, a simplified version of TDE (without multi-head) """

    def __init__(self, num_classes=10, in_dim=640, alpha=3., gamma=0, scale=16,
                 posthoc=False, bias=False, mu=0.9, **kwargs):
        super(TDESim_Classifier, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.scale = scale
        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.posthoc = posthoc
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.moving_ed = Parameter(torch.Tensor(1, in_dim).cuda(), requires_grad=False)

        self.init_weights()
        print(">> Feature Disentangling Classifier built!")

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):

        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True) + self.gamma)

        if self.posthoc and not self.training:
            # only at inference stage
            cos_val, sin_val = self.get_cos_sin(x, self.moving_ed)
            x = torch.mm(self.scale * (ex -  self.alpha * self.moving_ed * cos_val), ew.t()) + self.bias
        else:
            x = self.scale * (torch.mm(ex, ew.t())) + self.bias
            # moving average, do not record it at validation time during training.
            self.moving_ed.data = self.moving_ed.data * self.mu + torch.mean(ex, dim=0)
        return x

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

class CosPlus_Classifier(nn.Module):
    """ class of basic cosine classifier with more features for flexible adjustments """

    def __init__(self, num_classes=10, in_dim=640,
                 scale=16, bias=False, gamma=0.03125, eta=1,
                 moving_avg=True, mu=0.9, **kwargs):
        super(CosPlus_Classifier, self).__init__()
        self.num_classes = num_classes
        self.moving_avg = moving_avg
        self.in_dim = in_dim
        self.scale = scale
        self.gamma = gamma
        self.eta = eta
        self.mu = mu

        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        if self.moving_avg:
            self.moving_ed = Parameter(torch.Tensor(1, in_dim).cuda(), requires_grad=False)

        self.init_weights()
        print(">> CosPlus Classifier built!")

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):

        if self.eta != 1:
            ex = x / torch.norm(x.clone(), 2, 1, keepdim=True) ** self.eta
            ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True)** self.eta + self.gamma)
        else:
            ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
            ew = self.weight / (torch.norm(self.weight, 2, 1, keepdim=True) + self.gamma)
        x = self.scale * (torch.mm(ex, ew.t())) + self.bias

        if self.training and self.moving_avg:
            # record moving average
            self.moving_ed.data = self.moving_ed.data * self.mu + torch.mean(ex, dim=0)

        return x

class CDT_Classifier(nn.Module):
    """ class of Class-Dependent Temperatures loss """

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None, tau=1, bias=False, posthoc=False):
        super(CDT_Classifier, self).__init__()
        self.posthoc = posthoc
        if self.posthoc:
            self.nc_tau = torch.tensor(np.asarray(samples_per_cls) ** tau).float().cuda()
            print("Post-hoc re-scaling during testing, with tau={}".format(tau))
            # print("nc_tau:", self.nc_tau)
        else:
            self.ac_tau = torch.tensor((np.max(samples_per_cls) / samples_per_cls) ** tau).float().cuda()
            print("Class-Dependent Temperatures during training, with tau={}".format(tau))
            # print("ac_tau:", self.ac_tau)
        self.fc = nn.Linear(in_dim, num_classes)

        self.fc.bias.requires_grad = bias
        self.bias = bias

        print(">> CDT Classifier built!")

    def forward(self, x, **kwargs):
        # load plain-train model and add class-dependent temperature during testing
        if self.posthoc:
            x = self.fc(x) / self.nc_tau
        # training with class-dependent temperature
        elif self.training:
            x = self.fc(x) / self.ac_tau
        # trained with temperature, and test without it
        else:
            x = self.fc(x)
        return x

class PostNorm_Classifier(nn.Module):
    """ perform classifier weight normalization, also used for Learnable Weight Scaling (LWS) """

    def __init__(self, num_classes=10, in_dim=640, norm=False, feature_norm=False, lws=False, tau=0, bias=False, avg_T=1):
        super(PostNorm_Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.weight_norm = norm
        self.feature_norm = feature_norm
        self.tau = tau
        self.bias = bias
        self.avg_T = avg_T
        self.cla_scale = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=lws)
        self.init_weights()

        print(">> normalized classifier built, with weight_norm={}, feature_norm={}, uniform temperature avg_T={}".format(
            self.weight_norm, self.feature_norm, self.avg_T))

    def init_weights(self):
        self.cla_scale.data.fill_(1.)

    def forward(self, x, **kwargs):
        if self.weight_norm:
            ew = self.fc.weight / torch.pow(torch.norm(self.fc.weight.clone().detach(), 2, 1, keepdim=True), self.tau)
        else:
            ew = self.fc.weight
        if self.feature_norm:
            x = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        x = torch.mm(x, ew.t()) / (self.avg_T) * self.cla_scale
        if self.bias:
            x += self.fc.bias / (self.avg_T)
        return x

class Cos_Center_Classifier(nn.Module):
    """ cosine classifier based on the feature center of the training set """

    def __init__(self,  num_classes=10, in_dim=640):
        super(Cos_Center_Classifier, self).__init__()

        self.num_classes = num_classes
        self.in_dim = in_dim
        self.process_train = True

    def process_train_data(self, train_data):
        print(">> estimating center from saved training data features")
        labels = np.asarray(train_data['labels'])
        features = train_data['features']

        self.feature_means = torch.zeros(self.num_classes, self.in_dim).cuda()
        for i in range(self.num_classes):
            feature_cla = features[labels == i]
            self.feature_means[i] = torch.tensor(np.mean(feature_cla[0], axis=0)).cuda()


    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.feature_means / torch.norm(self.feature_means, 2, 1, keepdim=True)
        out = torch.mm(ex, ew.t())
        return out

if __name__ == '__main__':
   pass

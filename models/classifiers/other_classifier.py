import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import os
import math
import pdb
import time
import copy
import mmcv
from sklearn.cluster import KMeans


# This file contains implementations of some algorithms not reported in the paper,
# which we think could be of close relationship to the task.
# Since they have not been double checked or cleaned up, feel free to contact
# us should you have any questions.

class BN_Classifier(nn.Module):

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None, norm=True,
                 use_tracked_mean=True, use_tracked_var=True, use_relu=True,
                 bn_names=('CLEAN', 'PGD-5', 'PGD-10', 'PGD-20'),
                 bn_per_cls=False):
        super(BN_Classifier, self).__init__()

        self.num_classes = num_classes
        self.samples_per_cls = samples_per_cls
        self.in_dim = in_dim

        self.fc = nn.Linear(in_dim, num_classes)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)

        self.norm = norm
        self.use_relu = use_relu
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var
        self.bn_per_cls = bn_per_cls

        self.bn_names = bn_names
        self.bn_pool = {}
        self.init_bn_pool()

    def init_bn_pool(self):
        if self.bn_per_cls:
            self.bn_running_means = nn.Parameter(torch.zeros(len(self.bn_names), self.num_classes, self.in_dim), requires_grad=False).cuda()
            self.bn_running_vars = nn.Parameter(torch.ones(len(self.bn_names), self.num_classes, self.in_dim), requires_grad=False).cuda()
            for idx, name in enumerate(self.bn_names):
                self.bn_pool.update(
                {name: [nn.BatchNorm2d(self.in_dim).cuda() for _ in range(self.num_classes)]}
                )
                for i in range(self.num_classes):
                    self.bn_running_means[idx][i] = self.bn_pool[name][i].running_mean
                    self.bn_running_vars[idx][i] = self.bn_pool[name][i].running_var
        else:
            self.bn_running_means = nn.Parameter(torch.zeros(len(self.bn_names), self.in_dim), requires_grad=False).cuda()
            self.bn_running_vars = nn.Parameter(torch.ones(len(self.bn_names), self.in_dim), requires_grad=False).cuda()
            for idx, name in enumerate(self.bn_names):
                self.bn_pool.update(
                    {name: nn.BatchNorm2d(self.in_dim).cuda()}
                )
                self.bn_running_means[idx] = self.bn_pool[name].running_mean
                self.bn_running_vars[idx] = self.bn_pool[name].running_var

        # self.name2idx = dict()
        # for idx, name in enumerate(self.bn_names):
        #     self.name2idx.update({name: idx})

    def fine_bn(self, inputs, targets, name):

        if self.bn_per_cls:
            for i in range(self.num_classes):
                self.bn_pool[name][i](inputs[targets == i])
        else:
            self.bn_pool[name](inputs)
        return

    def forward(self, inputs, targets=None, name=None):

        # separate bn
        if name is not None and self.training:
            self.fine_bn(inputs, targets, name)

        # overall bn
        out = self.bn1(inputs)
        if self.use_relu:
            out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_dim)
        out = self.fc(out)

        return out


    def on_epoch(self):
        if self.bn_per_cls:
            for idx, name in enumerate(self.bn_names):
                for i in range(self.num_classes):
                    self.bn_running_means[idx][i] = self.bn_pool[name][i].running_mean
                    self.bn_running_vars[idx][i] = self.bn_pool[name][i].running_var
        else:
            for idx, name in enumerate(self.bn_names):
                self.bn_running_means[idx] = self.bn_pool[name].running_mean
                self.bn_running_vars[idx] = self.bn_pool[name].running_var


class Mix_Classifier(nn.Module):

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None,
                 norm=True, alpha=1.0, noise_alpha=0.2, beta=0.9999):
        super(Mix_Classifier, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.noise_alpha = noise_alpha
        self.samples_per_cls = torch.tensor(samples_per_cls).cuda()
        self.fc = nn.Linear(in_dim, num_classes)
        self.in_dim = in_dim
        self.norm = norm

        effective_num = 1.0 - np.power(beta, samples_per_cls)

        self.effective_num = torch.tensor(effective_num).float().unsqueeze(0).cuda()
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weights = torch.tensor(weights).float().unsqueeze(0).cuda()
        print(self.weights)

    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        batch_size = x.shape[0]
        if self.alpha > 0:
            lam = torch.rand(batch_size,1,1,1).cuda()
        else:
            lam = 1

        lam = 1. - self.noise_alpha * (1 - lam) # lam itself is the bigger one

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        y_a = y
        y_b = y[index]

        # lam = torch.where(self.samples_per_cls[y_a] > self.samples_per_cls[y_b], lam, 1-lam)
        # lam = torch.where(self.samples_per_cls[y_a] == self.samples_per_cls[y_b], lam, lam/self.noise_alpha)
        lam = lam.view(-1,1,1,1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        lam = 1 - (1 - lam.view(-1)) * self.weights.squeeze()[y_b]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):

        loss = lam * nn.CrossEntropyLoss()(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss()(pred, y_b)

        return loss.mean()

    def forward (self, inputs, **kwargs):
        outputs = self.fc(inputs)
        return outputs

    def loss(self, logits, targets_a, targets_b, lam):
        loss = self.mixup_criterion(logits, targets_a, targets_b, lam)
        return loss


class MC_Classifier(nn.Module):

    def __init__(self, num_classes=10, in_dim=640, samples_per_cls=None,
                 num_centroids=2, norm=True):
        super(MC_Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_centroids = num_centroids
        self.samples_per_cls = samples_per_cls
        self.fc = nn.Linear(in_dim, num_classes)
        self.in_dim = in_dim
        self.norm = norm

        self.feature_bank = [[] for _ in range(self.num_classes)]
        self.centroids = torch.zeros((num_classes, num_centroids, in_dim))
        print(self.fc.weight.type)


    def forward (self, inputs, **kwargs):
        x = self.fc(inputs)
        self.batch_features = inputs
        return x

    def adv_loss(self, inputs, labels):
        logits = torch.matmul(inputs, self.centroids_classifier.t()) # (batch_size, self.num_classes*self.num_centroids)

        if self.norm:
            inputs_norm = inputs.norm(dim=-1, p=2, keepdim=True)
            centroids_norm = self.centroids_classifier.norm(dim=-1, p=2, keepdim=True)
            logits = logits / inputs_norm / centroids_norm.t()

        logits = logits.view(inputs.shape[0], self.num_classes, self.num_centroids)

        max_logits = torch.max(logits, dim=-1)
        min_logits = torch.min(logits, dim=-1)
        index = torch.zeros_like(max_logits[0], dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        output = torch.where(index, min_logits[0], max_logits[0])

        # output = torch.mean(logits, dim=-1)
        adv_loss = F.cross_entropy(output, labels)
        return adv_loss

    def on_epoch(self):
        # update feature centroids
        for cla, features in enumerate(self.feature_bank):
            # fixme: if we use distributed training, the features would be stored on different divices
            assert len(features) > 0
            features = np.asarray(features)
            if features.shape[0] < self.num_centroids:
                for i in range(self.num_centroids):
                    feat_tensor = torch.tensor(features)
                    self.centroids[cla][i] = torch.mean(feat_tensor, dim=0)
            else:
                kmeans = KMeans(n_clusters=self.num_centroids, random_state=0).fit(features)
                for i in range(self.num_centroids):
                    feat_tensor = torch.tensor(features[kmeans.labels_==i])
                    self.centroids[cla][i] = torch.mean(feat_tensor, dim=0)
        self.centroids_classifier = self.centroids.view(self.num_classes*self.num_centroids, -1).cuda()

        # update classifier weight with centroids
        # gamma = 0.1
        # update_classifier =  torch.nn.Parameter(self.centroids.mean(dim=1).clone().detach(), requires_grad=True).cuda()
        # self.fc.weight = (self.fc.weight + gamma * self.centroids.mean(dim=1).clone().detach().cuda()) / (1 + gamma)
        self.feature_bank = [[] for _ in range(self.num_classes)]
        print('>>> update centroids by feature back')
        return

    def update_feature_bank(self, inputs, labels):
        for feature, label in zip(inputs, labels):
            self.feature_bank[label].append(feature.clone().detach().cpu().numpy())
        # for i in range(self.num_classes):
        #     print(len(self.feature_bank[i]), end=' ')
        # print()

    def loss(self, logits, labels):
        # todo: more compact
        loss = F.cross_entropy(logits, labels)
        return loss

if __name__ == '__main__':
    num_classes = 3
    samples_per_cls = [4,6,8]
    in_dim = 5
    batch_size = 6
    data = torch.randn([batch_size, in_dim, 14, 14])
    labels = torch.tensor([0,1,1,2,2,2], dtype=torch.int64)

    bn = BN_Classifier(num_classes, in_dim, samples_per_cls, bn_per_cls=True)
    bn(data, labels, 'CLEAN')

    num_centroids = 2
    mc = MC_Classifier(num_classes, in_dim, samples_per_cls, num_centroids)
    for i, spc in enumerate(samples_per_cls):
        for _ in range(spc):
            mc.feature_bank[i].append(torch.randn(in_dim).numpy())
    mc.on_epoch()
    # mc.centroids = torch.randn(num_classes, num_centroids, in_dim)
    feat_inputs = torch.randn(batch_size, in_dim)
    labels = torch.tensor([0,1,1,2,2,2], dtype=torch.int64)
    mc.adv_loss(feat_inputs, labels)






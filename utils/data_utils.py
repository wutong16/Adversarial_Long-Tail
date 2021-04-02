import logging
import os
import numpy as np
import torch
import random
import datetime
import csv
from mmcv.runner import get_dist_info
from mmcv import mkdir_or_exist
np.seterr(divide='ignore', invalid='ignore')


def save_csv(name, save_list, root='./data_ana', msg=True, devide=True):
    mkdir_or_exist(root)
    name = os.path.join(root, name)
    one_line = []
    for save_line in save_list:
        assert isinstance(save_line, list)
        one_line.extend(save_line)
        if devide:
            one_line.append(' ')
    with open(name, 'a+', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(one_line)
    if msg:
        print("Date written to {}".format(name))

class CountMeter(object):
    def __init__(self, num_classes, non_zero=True, save_raw=False):
        self.num_classes = num_classes
        self.non_zero = non_zero
        self.save_raw = save_raw
        self.reset()

    def reset(self):
        self.n = 0
        # count non-zero
        self.n_per_class = np.zeros(self.num_classes)
        self.sum_values = np.zeros(self.num_classes)
        self.avg_values = np.zeros(self.num_classes)
        if self.save_raw:
            self.raw_values = None

    def update(self, data, target=None):
        self.n += data.shape[0]

        # data (n,), target=(n,)
        if len(data.shape) == 1:
            assert target is not None
            self.sum_values[target] += data
            for i in range(self.num_classes):
                self.n_per_class[i] += (target == i).sum()

        # data (n, dim), target=(n, dim) / None
        else:
            self.sum_values += np.sum(data, dim=0)
            if target is None:
                self.n_per_class += np.sum(data>0, dim=0)
            else:
                self.n_per_class += np.sum(target, dim=0)

        if self.non_zero:
            self.avg_values = self.sum_values / self.n_per_class
        else:
            self.avg_values = self.sum_values / self.n

        if self.save_raw:
            if self.raw_values is None:
                self.raw_values = data
            else:
                self.raw_values = np.vstack((self.raw_values, data))

    def save_data(self):
        raise NotImplementedError


if __name__ == '__main__':
    header = ['model_dir', 'nat_acc', 'rob_acc',' ']
    class_no = np.arange(100).tolist()
    header.extend(class_no)
    header.append(' ')
    header.extend(class_no)
    save_csv('./CIFAR100_all_results.csv',  header)

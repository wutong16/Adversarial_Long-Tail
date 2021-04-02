import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .non_local import NonLocal_Direct
from .custom_activations import build_custom_activation
from .custom_norm import select_norm


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activation_name='relu'):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = build_custom_activation(activation_name)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = build_custom_activation(activation_name)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activation_name='relu'):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activation_name)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activation_name):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activation_name))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, use_relu=True, use_fc=True,
                 denoise=(), activation_name='relu', norm_type='BN', norm_power=0.2, use_pool=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation_name)
        # 1st sub-block
        # self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation_name)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation_name)
        # select norm to be used
        self.normlayer = select_norm(norm_type, norm_power=norm_power)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        # self.bn1 = self.normlayer(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # self.relu = build_custom_activation(activation_name)
        if use_fc:
            self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.use_relu = use_relu
        self.use_fc = use_fc
        self.use_pool = use_pool

        self.denoise = denoise
        if self.denoise:
            self.denoise1 = NonLocal_Direct(in_channels=nChannels[1])
            self.denoise2 = NonLocal_Direct(in_channels=nChannels[2])
            self.denoise3 = NonLocal_Direct(in_channels=nChannels[3])

        self.use_aux_bn = False
        if self.use_aux_bn:
            self.bn_aux = nn.BatchNorm2d(nChannels[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        if '1' in self.denoise:
            out, _ = self.denoise1(out)
        out = self.block2(out)
        if '2' in self.denoise:
            out, _ = self.denoise2(out)
        out = self.block3(out)
        if '3' in self.denoise:
            out, _ = self.denoise3(out)

        if self.use_relu:
            if self.use_aux_bn:
                out = self.relu(self.bn_aux(out))
            else:
                out = self.relu(self.bn1(out))

        if not self.use_pool:
            return out

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.use_fc:
            out = self.fc(out)
        return out

    def free_bn(self):
        self.apply(set_bn_train)

    def freeze_bn(self):
        self.apply(set_bn_eval)

    def reset_bn(self):
        self.apply(reset_bn)

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        # m.track_running_stats = False

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        # m.track_running_stats = False

def reset_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.reset_running_stats()
        m.track_running_stats = True
        # print(m.num_batches_tracked)


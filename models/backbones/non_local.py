import torch
from torch import nn
from torch.nn import functional as F


class OLTR_ModulatedAttLayer(nn.Module):

    def __init__(self, in_channels, reduction = 2, mode='embedded_gaussian'):
        super(OLTR_ModulatedAttLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size = 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        # self.fc_channel = nn.Linear(7 * 7 * self.in_channels, self.in_channels)
        # self.fc_selector = nn.Linear(7 * 7 * self.in_channels, 1)

        # self.triplet_loss = TripletLoss(margin=0.2)

        self.init_weights()

    def init_weights(self):
        msra_list = [self.g, self.theta, self.phi]
        for m in msra_list:
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()

    def embedded_gaussian(self, x):
        # embedded_gaussian cal self-attention, which may not strong enough
        batch_size = x.size(0)

        g_x = self.g(x.clone()).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x.clone()).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x.clone()).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)

        x_flatten = x.view(-1, 7 * 7 * self.in_channels)

        spatial_att = self.fc_spatial(x_flatten)
        spatial_att = spatial_att.softmax(dim=1)

        spatial_att = spatial_att.view(-1, 7, 7).unsqueeze(1)
        spatial_att = spatial_att.expand(-1, self.in_channels, -1, -1)

        final = spatial_att * mask + x

        return final, [x, spatial_att, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplemented("The code has not been implemented.")
        return output, feature_maps


class NonLocal_Direct(nn.Module):

    def __init__(self, in_channels, reduction = 1, mode='embedded_gaussian'):
        super(NonLocal_Direct, self).__init__()
        self.in_channels = in_channels
        # self.reduction = reduction
        self.inter_channels = in_channels
        self.mode = mode
        print('## Built non-local block of mode: {}, in_channels: {}!'.format(mode, in_channels))

        # self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        # self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        # self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)

        # for the use of denoising, g, theta and phi are all identity functions

        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size = 1, bias=False)

        # self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        # self.fc_channel = nn.Linear(7 * 7 * self.in_channels, self.in_channels)
        # self.fc_selector = nn.Linear(7 * 7 * self.in_channels, 1)

        # self.triplet_loss = TripletLoss(margin=0.2)

        self.init_weights()

    def init_weights(self):
        # msra_list = [self.g, self.theta, self.phi]
        # for m in msra_list:
        #     nn.init.kaiming_normal_(m.weight.data)
        #     m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()

    def embedded_gaussian(self, x):
        # embedded_gaussian cal self-attention, which may not strong enough
        batch_size = x.size(0)

        # g_x = self.g(self.x.clone()).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        # theta_x = self.theta(x.clone()).view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(x.clone()).view(batch_size, self.inter_channels, -1)

        g_x = x.clone().view(batch_size, -1, self.inter_channels)
        theta_x = g_x.clone()
        phi_x = g_x.clone()

        map_t_p = torch.matmul(theta_x, phi_x.permute(0, 2, 1))
        if self.mode == 'embedded_gaussian':
            mask_t_p = F.softmax(map_t_p, dim=-1) # (HxW, HxW)
        elif self.mode == 'dot_product':
            mask_t_p = F.softmax(map_t_p, dim=-1)
        else:
            raise NotImplemented("The code has not been implemented.")
        map_ = torch.matmul(mask_t_p, g_x) # (batch_size, HxW, dim)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)

        final = mask + x

        return final, [x, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplemented("The code has not been implemented.")
        return output, feature_maps


if __name__ == '__main__':
    in_channels = 5
    H = W = 7
    batch_size = 3
    input = torch.randn((batch_size, in_channels, H, W))
    block = NonLocal_Direct(in_channels=in_channels)
    output, _ = block(input)









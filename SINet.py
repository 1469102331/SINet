import math
import torch
import torch.nn as nn
import torch.nn.init
from mamba_ssm import Mamba

import torch.nn.functional as F
import numpy as np

import torch.nn.init as init
import scipy.linalg
import thops
from My_Model.refine import Refine


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d=1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 32+32
        # channel_split_num: 32

        self.split_len1 = channel_split_num  # 32  32
        self.split_len2 = channel_num - channel_split_num  # 32  96

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv
            x, logdet = self.flow_permutation(x, logdet=0, rev=rev)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            y = torch.cat([y1, y2], dim=1)
        else:
            #             print(x.shape)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = (x1 - self.F(y2))
            y = torch.cat((y1, y2), 1)
            y, logdet = self.flow_permutation(y, logdet=0, rev=rev)

        return y


class SpaBlock(nn.Module):
    def __init__(self, inchannels, flag=False):
        super(SpaBlock, self).__init__()
        self.flag = flag
        self.channels = inchannels
        self.dowm = HaarDownsampling(inchannels)
        self.operation1 = nn.Sequential(
            InvBlock(DenseBlock, inchannels, inchannels // 2),
            nn.LeakyReLU(0.2),
            InvBlock(DenseBlock, inchannels, inchannels // 2)
        )
        self.operation2 = nn.Sequential(
            InvBlock(DenseBlock, 4 * inchannels, inchannels),
            nn.LeakyReLU(0.2),
            InvBlock(DenseBlock, 4 * inchannels, inchannels)
        )

    def forward(self, x, rev=False):
        if self.flag:

            # 传递 rev 参数给 operation1 中的每个模块
            for module in self.operation1:
                if hasattr(module, 'forward') and callable(module.forward) and module.forward.__code__.co_argcount == 3:
                    x = module(x, rev)
                    # print(rev)
                else:
                    x = module(x)
        else:
            if not rev:
                x = self.dowm(x, rev=rev)

            for module in self.operation2:
                if hasattr(module, 'forward') and callable(module.forward) and module.forward.__code__.co_argcount == 3:
                    x = module(x, rev)
                else:
                    x = module(x)
            if rev:
                x = self.dowm(x, rev=rev)
        return x


class Net(nn.Module):
    def __init__(self, hschannels, mschannels, sf=8, bilinear=False, hidden_dim=32):
        super(Net, self).__init__()
        self.bilinear = bilinear
        self.sf = sf
        self.conv_hs1 = nn.Conv2d(hschannels, hidden_dim, 1, 1, 0)
        self.conv_ms = nn.Conv2d(mschannels, hidden_dim, 3, 1, 1)
        self.conv = nn.Conv2d(hschannels + mschannels, hidden_dim * 2, 1, 1, 0)
        self.ds = nn.Upsample(mode='bilinear', scale_factor=1 / 2)

        self.up2 = Upm(128 * 3, 64, bilinear)
        self.inc = nn.Sequential(
            BothMamba(hidden_dim * 4),
            FeedForward(hidden_dim * 2, 2 * hidden_dim),
        )
        self.fusion = nn.Sequential(
            BothMamba(128 * 4),
            Upm(256,128,bilinear)
        )
        self.down1 = Down(64 + 32 + 32, 32 * 4)
        self.down2 = Down(128 + 64 + 64, 128 * 2)
        self.outc = Refine(64 + 64 + 32, hschannels)

        self.spa1 = SpaBlock(hidden_dim, flag=True)
        self.spa2 = SpaBlock(hidden_dim)
        self.spa3 = SpaBlock(hidden_dim * 4)
        self.spa_2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=1, ),
            nn.LeakyReLU(0.2)
        )
        self.spa_3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 16, hidden_dim * 4, kernel_size=3, stride=1, padding=1, groups=1, ),
            nn.LeakyReLU(0.2)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spe0 = nn.Conv2d(hschannels, hidden_dim, 3, 1, 1)

        self.spe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.spe2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1, 1)
        self.spe3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 1, 1)

    def forward(self, hsi, msi):
        hsi = F.interpolate(hsi, scale_factor=self.sf, mode='bicubic', align_corners=False)  # 上采样
        spa0 = self.conv_ms(msi)  # 32 64 64
        spa1 = self.spa1(spa0)  # 32 64 64   --> 32 64 64
        spa2 = self.spa2(spa1)  # 32 64 64   --> 128 32 32
        spa_2 = self.spa_2(spa2)  #128 --> 64
        spa3 = self.spa3(spa2)  # 128 32 32 --> 512 16 16
        spa_3 = self.spa_3(spa3)  #512 --> 128

        spe0 = self.spe0(hsi)  # 31  64 64-> 32 64 64
        spe1 = self.spe1(spe0)  # 31 64 64 -> 32 64 64
        spe2 = self.spe2(self.avgpool(spe1))  # 32 64 64 -> 64 64 64
        spe3 = self.spe3(self.avgpool(spe2))  # 64 -> 128

        inv_spa2 = self.spa3(spa3, rev=True)  # 128*4 16 16 ->  128 32 32
        inv_spa1 = self.spa2(inv_spa2, rev=True)  # 128 32 32 -->32 64 64

        x = self.conv(torch.cat([hsi, msi], dim=1))
        x1 = self.inc(torch.cat([x, spa0, spe0], dim=1))  # 32 64 64 * 3 -->64 64 64
        x2 = self.down1(torch.cat([x1, spa1, spe1], dim=1))  # 64  + 32  +32  -->128
        x3 = self.down2(torch.cat([x2, spa_2, spe2], dim=1))  # C 128+ 64 +64 -->128*2 16 16

        x = self.fusion(torch.cat((x3, spa_3, spe3), dim=1)) # 128*2 + 128 + 128 16 16 -->128 32 32

        x = self.up2(torch.cat((x, inv_spa2, x2), dim=1))  # 128 32 32 *3 --> 32 64 64

        logits = self.outc(torch.cat((x, x1, inv_spa1), dim=1))  # 32+32+32

        return logits + hsi


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()

        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.proj = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):

        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape

        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x_proj + x
        else:
            return x_proj

class FeedForward(nn.Module):
    def __init__(self, dim, out_channels, mult=2, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * mult, out_channels, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # out = self.net(x.permute(0, 3, 1, 2))
        out = self.net(x)
        return out

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=1 / 2),
            BothMamba(in_channels),
            FeedForward(in_channels // 2, out_channels),
            # DoubleConv(in_channels // 3, out_channels),

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upm(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.maxpool_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                FeedForward(in_channels , out_channels),
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.PixelShuffle(2),
                FeedForward(in_channels//4, out_channels),
            )

    def forward(self, x1):
        return self.maxpool_conv(x1)


class HaarDownsampling(nn.Module):
    """C*H*W --> 4C*0.5H*0.5W"""

    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = int(channel_in)

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, c, rev=False):
        return self.last_jac


class ResConv(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x) + x


class AdaptiveGlobalLocalFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveGlobalLocalFusion, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2  , kernel_size=3, stride=1,
                                padding=1)

        self.ca = ChannelAttention(channels * 2  )
        self.spa_att = nn.Sequential(nn.Conv2d(channels * 2 , channels , kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels , channels * 2 , kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv_x(x)
        x = self.ca(x) * x
        x = self.spa_att(x) * x
        return (x)

class BothMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.half_channels = channels // 4  # 32  32
        self.use_residual = use_residual
        self.conv = nn.Conv2d(in_channels=channels // 2, out_channels= channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3 *  channels // 4, out_channels=channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3 *  channels // 4, out_channels=channels // 2, kernel_size=3, stride=1, padding=1)

        self.spa_mamba = SpaMamba( 2 * (channels // 4), use_residual=use_residual)

        self.spe_conv = ResConv( 2 * (channels // 4))
        self.ca = ChannelAttention(channels // 2)
        self.spa_att = nn.Sequential(nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())

        self.aglf = AdaptiveGlobalLocalFusion(channels//4)
    def forward(self, x):
        x, msi, hsi = x.narrow(1, 0, self.half_channels * 2), x.narrow(1, self.half_channels * 2,
                                                                       self.half_channels), x.narrow(
            1, self.half_channels * 3, self.half_channels)
        spa_x = self.conv1(torch.cat([msi, x], dim=1))
        spa_x = self.spa_mamba(spa_x)
        spe_x = self.conv2(torch.cat([hsi, x], dim=1))
        spe_x = self.spe_conv(spe_x)  #
        f0 = self.conv(x)
        sa = self.spa_att(spa_x)
        ca = self.ca(spe_x)
        f1 = sa * f0
        f1 = ca * f1
        fusion_x = self.aglf(spe_x,spa_x)
        return  fusion_x + f1






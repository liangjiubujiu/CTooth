# import torch.nn as nn
# import torch
# from torchsummary import summary
# from lib.medzoo.BaseModelClass import BaseModel
#
# """
# Implementation of this model is borrowed and modified
# (to support multi-channels and latest pytorch version)
# from here:
# https://github.com/Dawn90/V-Net.pytorch
# """
#
#
# def passthrough(x, **kwargs):
#     return x
#
#
# def ELUCons(elu, nchan):
#     if elu:
#         return nn.ELU(inplace=True)
#     else:
#         return nn.PReLU(nchan)
#
#
# class LUConv(nn.Module):
#     def __init__(self, nchan, elu):
#         super(LUConv, self).__init__()
#         self.relu1 = ELUCons(elu, nchan)
#         self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
#
#         self.bn1 = torch.nn.BatchNorm3d(nchan)
#
#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         return out
#
#
# def _make_nConv(nchan, depth, elu):
#     layers = []
#     for _ in range(depth):
#         layers.append(LUConv(nchan, elu))
#     return nn.Sequential(*layers)
#
#
# class InputTransition(nn.Module):
#     def __init__(self, in_channels, elu):
#         super(InputTransition, self).__init__()
#         self.num_features = 16
#         self.in_channels = in_channels
#
#         self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)
#
#         self.bn1 = torch.nn.BatchNorm3d(self.num_features)
#
#         self.relu1 = ELUCons(elu, self.num_features)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         repeat_rate = int(self.num_features / self.in_channels)
#         out = self.bn1(out)
#         x16 = x.repeat(1, repeat_rate, 1, 1, 1)
#         return self.relu1(torch.add(out, x16))
#
#
# class DownTransition(nn.Module):
#     def __init__(self, inChans, nConvs, elu, dropout=False):
#         super(DownTransition, self).__init__()
#         outChans = 2 * inChans
#         self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
#         self.bn1 = torch.nn.BatchNorm3d(outChans)
#
#         self.do1 = passthrough
#         self.relu1 = ELUCons(elu, outChans)
#         self.relu2 = ELUCons(elu, outChans)
#         if dropout:
#             self.do1 = nn.Dropout3d()
#         self.ops = _make_nConv(outChans, nConvs, elu)
#
#     def forward(self, x):
#         down = self.relu1(self.bn1(self.down_conv(x)))
#         out = self.do1(down)
#         out = self.ops(out)
#         out = self.relu2(torch.add(out, down))
#         return out
#
#
# class UpTransition(nn.Module):
#     def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
#         super(UpTransition, self).__init__()
#         self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
#
#         self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
#         self.do1 = passthrough
#         self.do2 = nn.Dropout3d()
#         self.relu1 = ELUCons(elu, outChans // 2)
#         self.relu2 = ELUCons(elu, outChans)
#         if dropout:
#             self.do1 = nn.Dropout3d()
#         self.ops = _make_nConv(outChans, nConvs, elu)
#
#     def forward(self, x, skipx):
#         out = self.do1(x)
#         skipxdo = self.do2(skipx)
#         out = self.relu1(self.bn1(self.up_conv(out)))
#         xcat = torch.cat((out, skipxdo), 1)
#         out = self.ops(xcat)
#         out = self.relu2(torch.add(out, xcat))
#         return out
#
#
# class OutputTransition(nn.Module):
#     def __init__(self, in_channels, classes, elu):
#         super(OutputTransition, self).__init__()
#         self.classes = classes
#         self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
#         self.bn1 = torch.nn.BatchNorm3d(classes)
#
#         self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
#         self.relu1 = ELUCons(elu, classes)
#
#     def forward(self, x):
#         # convolve 32 down to channels as the desired classes
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.conv2(out)
#         return out
#
#
# class Vnet(BaseModel):
#     """
#     Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
#     """
#
#     def __init__(self, elu=True, in_channels=1, classes=4):
#         super(Vnet, self).__init__()
#         self.classes = classes
#         self.in_channels = in_channels
#
#         self.in_tr = InputTransition(in_channels, elu=elu)
#         self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
#         self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
#         self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
#         self.up_tr64 = UpTransition(128, 64, 1, elu)
#         self.up_tr32 = UpTransition(64, 32, 1, elu)
#         self.out_tr = OutputTransition(32, classes, elu)
#
#     def forward(self, x):
#         out16 = self.in_tr(x)
#         out32 = self.down_tr32(out16)
#         out64 = self.down_tr64(out32)
#         out128 = self.down_tr128(out64)
#         out256 = self.down_tr256(out128)
#         out = self.up_tr256(out256, out128)
#         out = self.up_tr128(out, out64)
#         out = self.up_tr64(out, out32)
#         out = self.up_tr32(out, out16)
#         out = self.out_tr(out)
#         return out
#
#     def test(self,device='cpu'):
#         input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
#         ideal_out = torch.rand(1, self.classes, 32, 32, 32)
#         out = self.forward(input_tensor)
#         assert ideal_out.shape == out.shape
#         summary(self.to(torch.device(device)), (self.in_channels, 32, 32, 32),device=device)
#         # import torchsummaryX
#         # torchsummaryX.summary(self, input_tensor.to(device))
#         print("Vnet test is complete")
#
#
# class VNetLight(BaseModel):
#     """
#     A lighter version of Vnet that skips down_tr256 and up_tr256 in oreder to reduce time and space complexity
#     """
#
#     def __init__(self, elu=True, in_channels=1, classes=4):
#         super(VNetLight, self).__init__()
#         self.classes = classes
#         self.in_channels = in_channels
#
#         self.in_tr = InputTransition(in_channels, elu)
#         self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
#         self.up_tr64 = UpTransition(128, 64, 1, elu)
#         self.up_tr32 = UpTransition(64, 32, 1, elu)
#         self.out_tr = OutputTransition(32, classes, elu)
#
#     def forward(self, x):
#         out16 = self.in_tr(x)
#         out32 = self.down_tr32(out16)
#         out64 = self.down_tr64(out32)
#         out128 = self.down_tr128(out64)
#         out = self.up_tr128(out128, out64)
#         out = self.up_tr64(out, out32)
#         out = self.up_tr32(out, out16)
#         out = self.out_tr(out)
#         return out
#
#     def test(self,device='cpu'):
#         input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
#         ideal_out = torch.rand(1, self.classes, 32, 32, 32)
#         out = self.forward(input_tensor)
#         assert ideal_out.shape == out.shape
#         summary(self.to(torch.device(device)), (self.in_channels, 32, 32, 32),device=device)
#         # import torchsummaryX
#         # torchsummaryX.summary(self, input_tensor.to(device))
#
#         print("Vnet light test is complete")
#
#
# #m = VNet(in_channels=1,num_classes=2)
# #m.test()

import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Vnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(Vnet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out
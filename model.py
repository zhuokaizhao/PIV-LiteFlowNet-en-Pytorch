# model file of PIV-LiteFlowNet-en
# Author: Zhuokai Zhao

import os
import time
import torch
import datetime
import itertools
import numpy as np
from PIL import Image

import correlation
import layers

# use cudnn
torch.backends.cudnn.enabled = True

# PIV-LiteFlowNet-En (Cai)
class PIV_LiteFlowNet_en(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PIV_LiteFlowNet_en, self).__init__()

        # number of channels for input images
        self.num_channels = 1
        # magnitude change parameter of the flow
        self.backward_scale = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625]

        # feature descriptor (generate pyramidal features)
        class NetC(torch.nn.Module):
            def __init__(self, num_channels):
                super(NetC, self).__init__()

                # six-module (levels) feature extractor
                self.module_one = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=num_channels,
                                    out_channels=32,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_two = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # third conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_three = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_four = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_five = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_six = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=192,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, image_tensor):
                # generate six level features
                level_one_feat = self.module_one(image_tensor)
                level_two_feat = self.module_two(level_one_feat)
                level_three_feat = self.module_three(level_two_feat)
                level_four_feat = self.module_four(level_three_feat)
                level_five_feat = self.module_five(level_four_feat)
                level_six_feat = self.module_six(level_five_feat)

                return [level_one_feat,
                        level_two_feat,
                        level_three_feat,
                        level_four_feat,
                        level_five_feat,
                        level_six_feat]

        # matching unit
        class Matching(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Matching, self).__init__()

                self.flow_scale = backward_scale[level]

                if level == 2:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                else:
                    self.feature_net = torch.nn.Sequential()

                # No flow at the top level so no need to upsample
                if level == 6:
                    self.upsample_flow = None
                # up-sample the flow
                else:
                    self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=2,
                                                                  out_channels=2,
                                                                  kernel_size=4,
                                                                  stride=2,
                                                                  padding=1,
                                                                  bias=False,
                                                                  groups=2)

                # to speed up, no correlation on level 4, 5, 6
                if level >= 4:
                    self.upsample_corr = None
                # upsample the correlation
                else:
                    self.upsample_corr = torch.nn.ConvTranspose2d(in_channels=49,
                                                                  out_channels=49,
                                                                  kernel_size=4,
                                                                  stride=2,
                                                                  padding=1,
                                                                  bias=False,
                                                                  groups=49)

                self.matching_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                    stride=1,
                                    padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                )


            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # process feature tensors further based on levels
                feat_tensor_1 = self.feature_net(feat_tensor_1)
                feat_tensor_2 = self.feature_net(feat_tensor_2)

                # upsample and scale the current flow
                if self.upsample_flow != None:
                    flow_tensor = self.upsample_flow(flow_tensor)
                    flow_tensor_scaled = flow_tensor * self.flow_scale
                    # feature warping
                    feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)

                # level 4, 5, 6 it is None
                if self.upsample_corr == None:
                    # compute the corelation between feature 1 and warped feature 2
                    corr_tensor = correlation.FunctionCorrelation(tenFirst=feat_tensor_1, tenSecond=feat_tensor_2, intStride=1)
                    corr_tensor = torch.nn.functional.leaky_relu(input=corr_tensor, negative_slope=0.1, inplace=False)
                else:
                    # compute the corelation between feature 1 and warped feature 2
                    corr_tensor = correlation.FunctionCorrelation(tenFirst=feat_tensor_1, tenSecond=feat_tensor_2, intStride=2)
                    corr_tensor = torch.nn.functional.leaky_relu(input=corr_tensor, negative_slope=0.1, inplace=False)
                    corr_tensor = self.upsample_corr(corr_tensor)

                # put correlation into matching CNN
                delta_um = self.matching_cnn(corr_tensor)

                if flow_tensor != None:
                    return flow_tensor + delta_um
                else:
                    return delta_um

        # subpixel unit
        class Subpixel(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Subpixel, self).__init__()

                self.flow_scale = backward_scale[level]

                # same feature process as in Matching
                if level == 2:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                    # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                    self.normalize_flow = torch.nn.Conv2d(in_channels=2,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)
                else:
                    self.feature_net = torch.nn.Sequential()
                    # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                    self.normalize_flow = torch.nn.Conv2d(in_channels=2,
                                                            out_channels=32,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)


                # subpixel CNN that trains output to further improve flow accuracy
                self.subpixel_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 192, 160, 224, 288, 416 ][level],
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                    stride=1,
                                    padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                )




            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # process feature tensors further based on levels
                feat_tensor_1 = self.feature_net(feat_tensor_1)
                feat_tensor_2 = self.feature_net(feat_tensor_2)

                if flow_tensor != None:
                    # use flow from matching unit to warp feature 2 again
                    flow_tensor_scaled = flow_tensor * self.flow_scale
                    feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)

                # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                flow_tensor_scaled = self.normalize_flow(flow_tensor_scaled)

                # volume that is going to be fed into subpixel CNN
                volume = torch.cat([feat_tensor_1, feat_tensor_2, flow_tensor_scaled], axis=1)
                delta_us = self.subpixel_cnn(volume)

                return flow_tensor + delta_us

        class Regularization(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Regularization, self).__init__()

                self.flow_scale = backward_scale[level]

                self.unfold = [ 0, 0, 7, 5, 5, 3, 3 ][level]

                if level >= 5:
                    self.feature_net = torch.nn.Sequential()

                    self.dist_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                        stride=1,
                                        padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                    )
                else:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][level],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                    self.dist_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][level], 1),
                                        stride=1,
                                        padding=([ 0, 0, 3, 2, 2, 1, 1 ][level], 0)),

                        torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][level]),
                                        stride=1,
                                        padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][level]))
                    )


                # network that scales x and y
                self.scale_x_net = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                                    out_channels=1,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

                self.scale_y_net = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                                    out_channels=1,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

                self.regularization_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][level],
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )


            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # distance between feature 1 and warped feature 2
                flow_tensor_scaled = flow_tensor * self.flow_scale
                feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)
                squared_diff_tensor = torch.pow((feat_tensor_1 - feat_tensor_2), 2)
                # sum the difference in both x and y
                squared_diff_tensor = torch.sum(squared_diff_tensor, dim=1, keepdim=True)
                # take the square root
                # diff_tensor = torch.sqrt(squared_diff_tensor)
                diff_tensor = squared_diff_tensor

                # construct volume
                volume_tensor = torch.cat([ diff_tensor, flow_tensor - flow_tensor.view(flow_tensor.shape[0], 2, -1).mean(2, True).view(flow_tensor.shape[0], 2, 1, 1), self.feature_net(feat_tensor_1) ], axis=1)
                dist_tensor = self.regularization_cnn(volume_tensor)
                dist_tensor = self.dist_net(dist_tensor)
                dist_tensor = dist_tensor.pow(2.0).neg()
                dist_tensor = (dist_tensor - dist_tensor.max(1, True)[0]).exp()

                divisor_tensor = dist_tensor.sum(1, True).reciprocal()

                dist_tensor_unfold_x = torch.nn.functional.unfold(input=flow_tensor[:, 0:1, :, :],
                                                                  kernel_size=self.unfold,
                                                                  stride=1,
                                                                  padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)

                dist_tensor_unfold_y = torch.nn.functional.unfold(input=flow_tensor[:, 1:2, :, :],
                                                                  kernel_size=self.unfold,
                                                                  stride=1,
                                                                  padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)


                scale_x_tensor = divisor_tensor * self.scale_x_net(dist_tensor * dist_tensor_unfold_x)
                scale_y_tensor = divisor_tensor * self.scale_y_net(dist_tensor * dist_tensor_unfold_y)


                return torch.cat([scale_x_tensor, scale_y_tensor], axis=1)

        # combine all units
        self.NetC = NetC(self.num_channels)
        self.Matching = torch.nn.ModuleList([Matching(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])
        self.Subpixel = torch.nn.ModuleList([Subpixel(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])
        self.Regularization = torch.nn.ModuleList([Regularization(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])

        self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=2,
                                                        out_channels=2,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1,
                                                        bias=False,
                                                        groups=2)

    def forward(self, input_image_pair):
        image_tensor_1 = input_image_pair[:, 0:1, :, :]
        image_tensor_2 = input_image_pair[:, 1:, :, :]
        feat_tensor_pyramid_1 = self.NetC(image_tensor_1)
        feat_tensor_pyramid_2 = self.NetC(image_tensor_2)

        image_tensor_pyramid_1 = [image_tensor_1]
        image_tensor_pyramid_2 = [image_tensor_2]
        for level in [1, 2, 3, 4, 5]:
            # downsample image to match the different levels in the feature pyramid
            new_image_tensor_1 = torch.nn.functional.interpolate(input=image_tensor_pyramid_1[-1],
                                                                 size=(feat_tensor_pyramid_1[level].shape[2], feat_tensor_pyramid_1[level].shape[3]),
                                                                 mode='bilinear',
                                                                 align_corners=False)

            image_tensor_pyramid_1.append(new_image_tensor_1)

            new_image_tensor_2 = torch.nn.functional.interpolate(input=image_tensor_pyramid_2[-1],
                                                                 size=(feat_tensor_pyramid_2[level].shape[2], feat_tensor_pyramid_2[level].shape[3]),
                                                                 mode='bilinear',
                                                                 align_corners=False)

            image_tensor_pyramid_2.append(new_image_tensor_2)


        # initialize empty flow
        flow_tensor = None

        for level in [-1, -2, -3, -4, -5]:
            flow_tensor = self.Matching[level](flow_tensor,
                                               image_tensor_pyramid_1[level],
                                               image_tensor_pyramid_2[level],
                                               feat_tensor_pyramid_1[level],
                                               feat_tensor_pyramid_2[level])

            flow_tensor = self.Subpixel[level](flow_tensor,
                                               image_tensor_pyramid_1[level],
                                               image_tensor_pyramid_2[level],
                                               feat_tensor_pyramid_1[level],
                                               feat_tensor_pyramid_2[level])

            flow_tensor = self.Regularization[level](flow_tensor,
                                                     image_tensor_pyramid_1[level],
                                                     image_tensor_pyramid_2[level],
                                                     feat_tensor_pyramid_1[level],
                                                     feat_tensor_pyramid_2[level])

        # upsample flow tensor
        # flow_tensor = torch.nn.functional.interpolate(input=flow_tensor,
        #                                                 size=(flow_tensor.shape[2]*2, flow_tensor.shape[3]*2),
        #                                                 mode='bilinear',
        #                                                 align_corners=False)

        # PIV-LiteFlowNet-en uses deconv to upsample (change 1)
        flow_tensor = self.upsample_flow(flow_tensor)

        return flow_tensor
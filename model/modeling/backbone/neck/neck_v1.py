#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AdelaiDet-master 
@File    ：neck_v1.py
@IDE     ：PyCharm 
@Author  ：flysky
@Date    ：2022/12/19 19:51
不保留FPN结构，直接进行HIM
'''


import torch
from torch import nn
import torch.nn.functional as F
import math

from detectron2.modeling.backbone import Backbone
from detectron2.layers import Conv2d, ShapeSpec, get_norm

__all__ = ["MyFPN"]

class MyFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", num_repeats=2,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(MyFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        stage = [int(math.log2(stride)) for stride in strides]

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage[-1], stage[-1] + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        # build bifpn
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats):
            mid_channels = 192
            if i == 0:
                in_channels_list = in_channels_per_feature
            else:
                in_channels_list = [
                    self._out_feature_channels[name] for name in self._out_features
                ]
            self.repeated_bifpn.append(SingleMyFPN(
                in_channels_list, mid_channels, out_channels, stage, norm
            ))

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        feats = [bottom_up_features[f] for f in self.in_features]

        for bifpn in self.repeated_bifpn:
             feats = bifpn(feats)

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = feats[self._out_features.index(self.top_block.in_feature)]
            feats.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(feats)
        return {f: res for f, res in zip(self._out_features, feats)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class SingleMyFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, mid_channels, out_channels, strides, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super(SingleMyFPN, self).__init__()

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.strides = strides

        lateral_convs = []
        group_convs = []
        output_convs = []
        for idx in range(3):
            stage = strides[idx%3]
            in_channels = in_channels_list[idx%3]

            if idx < 3:
                # 特征压缩分组
                group_conv = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Conv2d(
                        mid_channels,
                        mid_channels,
                        kernel_size=1,
                        norm=get_norm(norm, mid_channels),
                        activation=nn.ReLU()),
                    Conv2d(
                        mid_channels,
                        mid_channels * mid_channels,
                        kernel_size=1)
                )
                self.add_module("group_conv{}".format(stage), group_conv)
                group_convs.append(group_conv)

                # 特征融合
                output_conv = Conv2d(
                    mid_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm == ""),
                    norm=get_norm(norm, out_channels),
                )
            else:
                output_conv = Conv2d(
                        mid_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=(norm == ""),
                        norm=get_norm(norm, out_channels))

            # 降维
            lateral_conv = Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                norm=get_norm(norm, mid_channels)
            )

            self.add_module("fpn_lateral{}_{}".format(stage, idx//len(strides)+1), lateral_conv)
            lateral_convs.append(lateral_conv)

            self.add_module("fpn_output{}_{}".format(stage, idx//len(strides)+1), output_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs
        self.group_convs = group_convs
        self.output_convs = output_convs

    def forward(self, feats):
        groups_fests = []
        for i in range(len(feats)):
            group_f = self.lateral_convs[i](feats[i])
            group_f = self.feature_group(group_f, self.group_convs[i])
            group_f = group_f.split((64, 64, 64), 1)
            groups_fests.append(group_f)

        results = []
        for idx, x in enumerate(zip(groups_fests[0], groups_fests[1], groups_fests[2])):
            p = []
            _, _, target_h, target_w = x[idx].shape
            for x_c in x:
                x_c = F.interpolate(x_c, size=(target_h, target_w), mode="nearest")
                p.append(x_c)
            p = torch.cat(p, dim=1)
            p = self.output_convs[idx](p)
            results.append(p)
        return results


    def feature_group(self, x, model):
        b, c, w, h = x.size()
        x_s = x.view(b, c, -1)
        m = model(x)
        m = m.view(b, c, c)
        m = m @ x_s
        y = m.view(b, c, w, h)
        return y


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

import math
import torch
from torch import nn
import torch.nn.modules as nn
import torch.nn.functional as F
# from detectron2.modeling.poolers import ROIPooler
# import torch_dct
import pywt
import numpy as np


def build_boundary_enhancement_module():
    return BoundaryEnhancementModule()

class BoundaryEnhancementModule(nn.Module):
    def __init__(self,):
        super().__init__()

        channels = 8
        mid_channels = 16

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.fusion_feats = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )

        # self.fcanet = MultiSpectralAttentionLayer(mid_channels, 56, 56, reduction=2, freq_sel_method='top16')

        self.prediction = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        for modules in [
            self.image_conv,
            self.fusion_feats,
            self.prediction,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, mask_features, original_images):
        map_scores = torch.sigmoid(mask_features)

        original_features = self.image_conv(original_images)

        fine_features = original_features*map_scores

        all_features = torch.cat([mask_features, fine_features], dim=1)

        fusion_feat = self.fusion_feats(all_features)

        cba_feat = self.cba(fusion_feat)

        mask_pred = self.prediction(cba_feat)

        return mask_pred

    def cba(self,x):
        B, C, H, W = x.size()
        b = x.view(B, C, -1)
        a = x.view(B, C, -1).transpose(1, 2)
        c = torch.matmul(b, a)
        c = torch.sigmoid(c)
        out = torch.matmul(a, c).transpose(1, 2).view(B, C, H, W)
        out = out * x
        return out

    def twin_cba(self, x):
        b1 = self.poolh(x).squeeze(-1)
        a1 = b1.transpose(1, 2)
        c1 = torch.sigmoid(torch.matmul(b1, a1))
        map_h = torch.matmul(a1, c1).transpose(1, 2).unsqueeze(-1)

        b2 = self.poolw(x).squeeze(-2)
        a2 = b2.transpose(1, 2)
        c2 = torch.sigmoid(torch.matmul(b2, a2))
        map_w = torch.matmul(a2, c2).transpose(1, 2).unsqueeze(-2)

        out = x * map_h.expand_as(x) * map_w.expand_as(x)
        return out

import math
import torch
from torch import nn
import torch.nn.modules as nn
import torch.nn.functional as F
# from detectron2.modeling.poolers import ROIPooler
# import torch_dct
import pywt
import numpy as np


def build_edge_guided_module():
    return EdgeGuidedModule()

class EdgeGuidedModule(nn.Module):
    def __init__(self,):
        super().__init__()

        channels = 16

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.prediction = nn.Sequential(
            nn.Conv2d(channels+1, channels+1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels+1),
            nn.ReLU(),
            nn.Conv2d(channels+1, 1, kernel_size=1, stride=1, padding=0),
        )
        for modules in [
            self.image_conv,
            self.prediction,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, ins_pred_list, original_images):

        # H, W = ins_pred_list[0].shape[-2]
        H, W = 0, 0
        for ins in ins_pred_list:
            if ins is not None:
                H, W =ins.shape[-2:]
        # assert H==0 and W==0, 'error!'
        if H==0 and W==0:
            print('error error')
        original_images = F.interpolate(original_images, size=(H,W), mode='bilinear', align_corners=True)
        original_features = self.image_conv(original_images)
        ins_egm_pred_list = []
        for idx, ins_pred in enumerate(ins_pred_list):
            if ins_pred is None:
                continue
            map_scores = torch.sigmoid(ins_pred)
            fine_feats = original_features[idx].unsqueeze(0) * map_scores.unsqueeze(1)
            all_features = torch.cat([ins_pred.unsqueeze(1), fine_feats], dim=1)
            attention_feats = self.cba(all_features)
            mask_pred = self.prediction(attention_feats).squeeze(1)
            ins_egm_pred_list.append(mask_pred)
        return ins_egm_pred_list

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








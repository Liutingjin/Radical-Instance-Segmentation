import torch
from torch.nn import functional as F
from torch import nn
import copy
import math

from adet.utils.comm import compute_locations, aligned_bilinear
from .boundary_enhancement_module import build_boundary_enhancement_module
# from adet.modeling.condinst.transformer import TwinTransformer

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
        bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        # if l < num_layers - 1:
        #     # out_channels x in_channels x 1 x 1
        #     weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
        #     bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        # else:
        #     # out_channels x in_channels x 1 x 1
        #     weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
        #     bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            # elif l == self.num_layers - 1:
            #     weight_nums.append(self.channels * 1)
            #     bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.boundary_enhancement_module = build_boundary_enhancement_module()
        # self.twintransformer = TwinTransformer(dim=8, depth=2, heads=4)

        # self.prediction = nn.Sequential(
        #     nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        # )
        # for modules in [
        #     self.prediction,
        # ]:
        #     for l in modules.modules():
        #         if isinstance(l, nn.Conv2d):
        #             torch.nn.init.normal_(l.weight, std=0.01)
        #             torch.nn.init.constant_(l.bias, 0)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            x = F.relu(x)
            # if i < n_layers - 1:
            #     x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances, images_edge=None
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            if images_edge!=None:
                images_edge = F.interpolate(images_edge, size=(H, W), mode='bilinear', align_corners=True)
                images_edge = images_edge[im_inds]
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 8, H, W)
        # print(mask_logits.shape)
        # LIMIT = 35
        # mask_logits1 = mask_logits[:LIMIT]
        # mask_logits1 = F.interpolate(mask_logits1, size=(80, 80), mode='bilinear', align_corners=True)
        # mask_logits1 = self.twintransformer(mask_logits1)
        # mask_logits1 = F.interpolate(mask_logits1, size=(H, W), mode='bilinear', align_corners=True)
        # mask_logits[:LIMIT] = mask_logits1

        # mask_logits1 = mask_logits.detach()
        # mask_logits = self.prediction(mask_logits)
        mask_logits = self.boundary_enhancement_module(mask_logits, images_edge)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        # mask_pre = aligned_bilinear(mask_pre, int(mask_feat_stride / self.mask_out_stride))
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None, images_edge=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, images_edge=images_edge
                )
                # pre_mask_scores = mask_pre.sigmoid()
                mask_scores = mask_logits.sigmoid()

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    loss_mask = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = loss_mask.mean()

                    # loss_mask = ECE_loss(mask_scores, gt_bitmasks)
                    # losses["loss_pre_mask"] = loss_pre_mask
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits  = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, images_edge=images_edge
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances

def ECE_loss(mask_scores, gt_bitmasks):
    """
    Args:
        mask_scores (Tensor): A tensor of shape (B, 1, H, W)
        gt_bitmasks (Tensor): A tensor of shape (B, 1, H, W)
    """
    gt_edges, gt_centers = get_edge_gt(gt_bitmasks)

    mask_losses1 = dice_coefficient(mask_scores, gt_bitmasks)
    mask_losses1 = mask_losses1.mean()

    # entirety_loss = F.binary_cross_entropy(mask_scores, gt_bitmasks, weight=gt_centers, reduction="mean")
    # mask_losses = entirety_loss + mask_losses1

    return mask_losses1

def get_edge_gt(gt_masks):
    import numpy as np
    import cv2

    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gt_masks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_targets = F.conv2d(gt_masks, laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    # boundary_targets = boundary_targets.squeeze(1)

    gt_masks[boundary_targets==1] = 5

    # for i in range(len(gt_masks)):
    #     b = np.array(gt_masks[i])
    #     b[b == 1.] = 255
    #     cv2.imshow('origsa', b[0])
    #     cv2.waitKey(0)  # ????????????
    #
    #     a = np.array(boundary_targets[i])
    #     a[a == 1.] = 255
    #     cv2.imshow('pic_name', a[0])
    #     cv2.waitKey(0)  # ????????????

    return boundary_targets, gt_masks

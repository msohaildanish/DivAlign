# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers.smooth_l1_loss import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.apply_cal = cfg.MODEL.ROI_BOX_HEAD.CAL
        self.cal_alpha = cfg.MODEL.ROI_BOX_HEAD.CAL_ALPHA
        self.apply_ral = cfg.MODEL.ROI_BOX_HEAD.RAL
        self.ral_beta = cfg.MODEL.ROI_BOX_HEAD.RAL_BETA

    def kl_categorical(self, p_logit, q_logit, weights=None):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
        if weights is not None:
            _kl *= weights
        return torch.mean(_kl)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        device = features[0].device
        loss_cal = torch.tensor(0.0).to(device)
        loss_ral = torch.tensor(0.0).to(device)

        if self.training and (self.apply_cal or self.apply_ral):
            B = len(proposals)
            lengths = [len(x) for x in proposals]
            src_indices = torch.tensor(list(range(0, B, 2))).to(device)
            tgt_indices = src_indices + 1
            src_propoals = [proposals[i] for i in range(0, B, 2)]
            tgt_feats = [features[0][tgt_indices]] # NO FPN
            kl_x = self.feature_extractor(tgt_feats, src_propoals)
            kl_class_logits, kl_box_regression = self.predictor(kl_x)
        if self.training and self.apply_cal:
            class_logits_chunks = class_logits.split(lengths)
            src_logits = torch.cat([class_logits_chunks[i] for i in range(0, B, 2)])
            loss_cal = self.kl_categorical(kl_class_logits, src_logits)
        if self.training and self.apply_ral:
            box_regression_chunks = box_regression.split(lengths)
            src_regres = torch.cat([box_regression_chunks[i] for i in range(0, B, 2)])
            labels = cat([proposal.get_field("labels") for proposal in src_propoals], dim=0)
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)
            pred_bbox = kl_box_regression[sampled_pos_inds_subset[:, None], map_inds]
            src_bbox = src_regres[sampled_pos_inds_subset[:, None], map_inds]
            loss_ral = smooth_l1_loss(pred_bbox, src_bbox)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_cal=loss_cal*self.cal_alpha, loss_ral=loss_ral*self.ral_beta),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
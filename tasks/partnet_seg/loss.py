import torch
from torch import nn
from utils import loss_utils
from tasks.base.loss import LossFunc_base


def get_loss(name):
    if name == 'baseline':
        return LossBase
    elif name == 'joint':
        return LossJoint
    else:
        raise NotImplementedError


class LossBase(LossFunc_base):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def _compute_cls_loss(self, predict, target, weights):
        c = predict.shape[-1]
        predict = predict.view(-1, c).contiguous() # B * N, C
        target = target.view(-1).contiguous()  # B * N
        weights = weights.view(-1).contiguous() # B * N

        loss = self._cross_entropy_loss(predict, target) # B * N
        loss *= weights
        loss = torch.mean(loss)
        return loss

    def forward(self, input_dict, pred_dict):
        predict = pred_dict['pred_cls']
        labels = input_dict['seg_labels'].long().cuda()
        weights = input_dict['label_weights'].float().cuda()

        return self._compute_cls_loss(predict, labels, weights)


class LossJoint(LossBase):
    def _comput_shape_loss(self, pred_points, gt_points, hint_points=None):
        if self._cfgs.use_cd and self._cfgs.use_emd:
            # a * cd + (1 - a) * emd
            shape_loss_cd = loss_utils._cd_loss(pred_points, gt_points, alpha=self._cfgs.cd_alpha)
            shape_loss_emd = loss_utils._emd_loss_with_hint(pred_points, gt_points, hint_points)
            shape_loss = self._cfgs.cd_beta * shape_loss_cd + (1 - self._cfgs.cd_beta) * shape_loss_emd
        elif self._cfgs.use_cd:
            # a * f + (1 - a) * b
            shape_loss = loss_utils._cd_loss_with_hint(pred_points, gt_points, hint=hint_points, alpha=self._cfgs.cd_alpha)
        elif self._cfgs.use_emd:
            shape_loss = loss_utils._emd_loss_with_hint(pred_points, gt_points, hint_points)
        return shape_loss

    def update(self, epoch):
        if epoch > 0 and epoch % self._cfgs.beta_decay_step == 0:
            self._cfgs.beta *= self._cfgs.beta_decay
            print(' <loss.update> -- decay beta {:.3f} at epoch {}.'.format(self._cfgs.beta, epoch))

    def forward(self, input_dict, pred_dict):
        # for segmentation loss
        predict = pred_dict['pred_cls']
        labels = input_dict['seg_labels'].long().cuda()
        weights = input_dict['label_weights'].float().cuda()

        # for shape loss
        pred_points = pred_dict['pred_points']
        hint_points = pred_dict['hint_points']
        if not self._cfgs.use_hint:
            hint_points = None
        prefps_points = input_dict['prefps_points'].float().cuda()

        return self._cfgs.beta * self._comput_shape_loss(pred_points, prefps_points, hint_points) + \
            self._compute_cls_loss(predict, labels, weights)

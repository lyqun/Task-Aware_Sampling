from tasks.base.metric import Metric_base
import numpy as np


def get_metrics(split='train'):
    if split == 'train':
        return [Accuracy()]
    elif split in ['val', 'eval', 'test']:
        return [Accuracy(), Part_mIoU(), Shape_mIoU()]
    else:
        raise NotImplementedError


class Accuracy(Metric_base):
    def __init__(self):
        super().__init__(name='accuracy', is_min=False)
        self._tmp_vals = 0
        self._total_cnt = 0

    def eval_reset(self):
        self._tmp_vals = 0
        self._total_cnt = 0

    def eval_bs(self, input_dict, pred_dict):
        seg_probs = pred_dict['pred_cls'].data.cpu().numpy()
        seg_gt = input_dict['seg_labels'].data.cpu().numpy()
        seg_res = np.argmax(seg_probs[..., 1:], axis=-1) + 1

        self._tmp_vals += np.sum(np.mean((seg_res == seg_gt) | (seg_gt == 0), axis=-1))
        self._total_cnt += seg_gt.shape[0]
        
    def eval_detach(self):
        accuracy = self._tmp_vals / self._total_cnt
        self.eval_reset()
        return accuracy


class Part_mIoU(Metric_base):
    def __init__(self):
        super().__init__(name='part_miou', is_min=False)
        self._tmp_inter = None
        self._tmp_union = None

    def eval_reset(self):
        self._tmp_inter = None
        self._tmp_union = None

    def eval_bs(self, input_dict, pred_dict):
        num_class = input_dict['num_class'].data.cpu().numpy()[0]
        seg_probs = pred_dict['pred_cls'].data.cpu().numpy()
        seg_gt = input_dict['seg_labels'].data.cpu().numpy()
        seg_res = np.argmax(seg_probs[..., 1:], axis=-1) + 1
        seg_res[seg_gt == 0] = 0 # ignore points labeled as 0 (others/unlabeled)
        
        if self._tmp_inter is None:
            self._tmp_inter = np.zeros(num_class, dtype=np.float32)
            self._tmp_union = np.zeros(num_class, dtype=np.float32)

        for pred, gt in zip(seg_res, seg_gt):
            for i in range(1, num_class):
                gt_mask = gt == i
                pred_mask = pred == i

                self._tmp_inter[i] += np.sum(gt_mask & pred_mask)
                self._tmp_union[i] += np.sum(gt_mask | pred_mask)
        
    def eval_detach(self):
        part_ious = self._tmp_inter[1:] / (self._tmp_union[1:] + 1e-12)
        part_miou = np.mean(part_ious)
        self.eval_reset()
        return part_miou
    

class Shape_mIoU(Metric_base):
    def __init__(self):
        super().__init__(name='shape_miou', is_min=False)
        self._tmp_vals = []

    def eval_reset(self):
        self._tmp_vals = []

    def eval_bs(self, input_dict, pred_dict):
        num_class = input_dict['num_class'].data.cpu().numpy()[0]
        seg_probs = pred_dict['pred_cls'].data.cpu().numpy()
        seg_gt = input_dict['seg_labels'].data.cpu().numpy()
        seg_res = np.argmax(seg_probs[..., 1:], axis=-1) + 1
        seg_res[seg_gt == 0] = 0 # ignore points labeled as 0 (others/unlabeled)

        for pred, gt in zip(seg_res, seg_gt):
            shape_ious = []
            for i in range(1, num_class):
                gt_mask = gt == i
                pred_mask = pred == i
                if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
                    inter = np.sum(gt_mask & pred_mask)
                    union = np.sum(gt_mask | pred_mask)
                    shape_ious.append(inter / (union + 1e-12))

            if len(shape_ious) > 0:
                shape_miou = np.mean(shape_ious)
                self._tmp_vals.append(shape_miou)

    def eval_detach(self):
        shape_miou = np.mean(self._tmp_vals)
        self.eval_reset()
        return shape_miou



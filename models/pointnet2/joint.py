import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils

import importlib
from models.pointnet2.baseline import PointNet2_base


def get_model(cfgs):
    return PointNet2_joint(cfgs)

class PointNet2_joint(PointNet2_base):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        SAMPLER = importlib.import_module('samplers.' + self._cfgs.sampler.name)
        self.samplenet = SAMPLER.get_model(input_channels=self._cfgs.MLPS[0][-1], cfgs=self._cfgs.sampler)
        self._debug_flag = True

    def get_sampler_params(self):
        return self.samplenet.parameters()

    def get_pointnet2_params(self):
        sampler_params_ids = list(map(id, self.get_sampler_params()))
        return filter(lambda p: \
            id(p) not in sampler_params_ids, self.parameters())

    def forward(self, input_dict):
        pointcloud = input_dict['pointcloud'].float().cuda()
        xyz, features = self._break_up_pc(pointcloud)

        # downsampling
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.sa_modules)):
            new_xyz = None
            # learnable sampler
            if i == 1:
                # input pcd with features
                in_feats = torch.cat([l_xyz[-1], l_features[-1].transpose(1, 2)], axis=2)

                # hint points
                hint_idx = pointnet2_utils.furthest_point_sample(l_xyz[-1], self.npoints[1])
                hint_points = pointnet2_utils.gather_operation(
                    l_xyz[-1].transpose(1, 2).contiguous(), hint_idx) # B, 3, N'
                hint_xyz = hint_points.transpose(1, 2).contiguous()

                pred_points = self.samplenet(in_feats, hint_xyz=hint_xyz)
                new_xyz = pred_points
                if self._debug_flag:
                    print(' <model.forward> -- joint points (learned) {} at layer 1(2).'.format(new_xyz.shape))
                    self._debug_flag = False
            
            new_xyz = None
            li_xyz, li_features = self.sa_modules[i](l_xyz[i], l_features[i], new_xyz=new_xyz)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # upsampling
        for i in range(-1, -(len(self.fp_modules) + 1), -1):
            l_features[i - 1] = self.fp_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        # linear layer
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_class)
        return {
            'pred_cls': pred_cls,
            'hint_points': hint_xyz,
            'pred_points': pred_points
        }

import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule
import pointnet2.pytorch_utils as pt_utils

def get_model(cfgs):
    return PointNet2_base(cfgs)

class PointNet2_base(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self._cfgs = cfgs
        print(' <model> -- npoints:', self._cfgs.NPOINTS)
        print(' <model> -- radius:', self._cfgs.RADIUS)
        # print(self._cfgs)

        self.npoints = self._cfgs.NPOINTS
        input_channels = self._cfgs.extra_channels

        # initilize sa modules
        self.sa_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [input_channels]
        for k in range(len(self._cfgs.NPOINTS)):
            conv_mlp = [channel_in] + self._cfgs.MLPS[k]
            channel_out = conv_mlp[-1]

            self.sa_modules.append(
                PointnetSAModule(
                    npoint=self._cfgs.NPOINTS[k],
                    radius=self._cfgs.RADIUS[k],
                    nsample=self._cfgs.NSAMPLE[k],
                    mlp=conv_mlp,
                    use_xyz=True,
                    bn=self._cfgs.use_bn))
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        # initialize fp modules
        self.fp_modules = nn.ModuleList()
        for k in range(len(self._cfgs.FP_MLPS)):
            pre_channel = self._cfgs.FP_MLPS[k + 1][-1] if k + 1 < len(self._cfgs.FP_MLPS) else channel_out
            mlp = [pre_channel + skip_channel_list[k]] + self._cfgs.FP_MLPS[k]
            print(' <model> --', mlp)
            self.fp_modules.append(
                PointnetFPModule(mlp=mlp))

        # initilize linear layer for point-wise classification
        cls_layers = []
        pre_channel = self._cfgs.FP_MLPS[0][-1]
        for k in range(len(self._cfgs.CLS_FC)):
            cls_layers.append(pt_utils.Conv1d(pre_channel, self._cfgs.CLS_FC[k], bn=True))
            pre_channel = self._cfgs.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, self._cfgs.num_class, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, input_dict):
        pointcloud = input_dict['pointcloud'].float().cuda()
        xyz, features = self._break_up_pc(pointcloud)

        # downsampling
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.sa_modules)):
            li_xyz, li_features = self.sa_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # upsampling
        for i in range(-1, -(len(self.fp_modules) + 1), -1):
            l_features[i - 1] = self.fp_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        # linear layer
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_class)
        return {
            'pred_cls': pred_cls
        }

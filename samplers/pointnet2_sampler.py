import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule


def get_model(input_channels, cfgs):
    return Sampler_base(input_channels, cfgs)

class Sampler_base(nn.Module):
    def __init__(self, input_channels, cfgs):
        super().__init__()
        self._cfgs = cfgs

        print(' <sampler> -- npoints', self._cfgs.NPOINTS)
        print(' <sampler> -- radius', self._cfgs.RADIUS)
        
        self.npoints = self._cfgs.NPOINTS

        # downsample layer
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [channel_in]
        for k in range(len(self._cfgs.NPOINTS)):
            mlps = [self._cfgs.MLPS[k].copy()]
            channel_out = 0

            for idx in range(len(mlps)):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self._cfgs.NPOINTS[k],
                    radius=self._cfgs.RADIUS[k],
                    nsample=self._cfgs.NSAMPLE[k],
                    mlp=mlps[0],
                    use_xyz=True,
                    use_leaky=self._cfgs.use_leaky,
                    alpha=self._cfgs.leaky_alpha,
                    bn=self._cfgs.use_bn))
            
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()
        for k in range(1, len(self._cfgs.FP_MLPS)):
            pre_channel = self._cfgs.FP_MLPS[k + 1][-1] if k + 1 < len(self._cfgs.FP_MLPS) else channel_out
            mlp = [pre_channel + skip_channel_list[k]] + self._cfgs.FP_MLPS[k]
            self.FP_modules.append(
                PointnetFPModule(mlp=mlp, bn=self._cfgs.use_bn))
        
        channel_out = self._cfgs.FP_MLPS[1][-1] + 3
        self.loc_regressor = nn.Conv1d(in_channels=channel_out, out_channels=3, 
            kernel_size=1, bias=True)
        
        if self._cfgs.soft_proj:
            from pointnet2.soft_projection import SoftProjection
            self.project = SoftProjection(group_size=8, initial_temperature=1.0, 
                                      is_temperature_trainable=True, min_sigma=1e-2)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud, hint_xyz, no_proj=False):
        '''
        pointcloud: (B, N, 3)
        hint_xyz: (B, N', 3)
        '''
        xyz, features = self._break_up_pc(pointcloud)
        points = xyz.transpose(1, 2).contiguous() # B, 3, N
        hint_points = hint_xyz.transpose(1, 2).contiguous() # B, 3, N'

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            new_xyz = hint_xyz if i == 0 else None
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], new_xyz=new_xyz)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        ret_feature = l_features[1]
        ret_feature = torch.cat([hint_points, ret_feature], axis=1)
        pred_points = self.loc_regressor(ret_feature) # B, 3, N'

        if not self._cfgs.learn_loc: # learn offset
            pred_points = hint_points + pred_points

        if self._cfgs.soft_proj and not no_proj:
            pred_points = self.project(point_cloud=points, 
                    query_cloud=pred_points).contiguous()
        
        return pred_points.transpose(1, 2).contiguous() # B, N', 3


if __name__ == '__main__':
    from utils.config import load_config
    cfgs = load_config('./utils/configs/partnet.yaml')

    model = get_model(64, cfgs.model.sampler).cuda()
    input_pcd = torch.randn(2, 4096, 67).float().cuda()
    hint_xyz = torch.randn(2, 1024, 3).float().cuda()
    
    output = model(input_pcd, hint_xyz=hint_xyz)
    print(output.shape)
    
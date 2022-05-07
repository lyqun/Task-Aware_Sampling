from models.pointnet2.baseline import PointNet2_base

def get_model(cfgs):
    return PointNet2_prefps(cfgs)

class PointNet2_prefps(PointNet2_base):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self._debug_flag = True
    
    def forward(self, input_dict):
        pointcloud = input_dict['pointcloud'].float().cuda()
        prefps_points = input_dict['prefps_points'].float().cuda()
        xyz, features = self._break_up_pc(pointcloud)

        # downsampling
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.sa_modules)):
            new_xyz = None
            if i == 1:
                new_xyz = prefps_points
                if self._debug_flag:
                    print(' <model.forward> -- prefps points {} at layer 1(2).'.format(new_xyz.shape))
                    self._debug_flag = False
            
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
            'pred_cls': pred_cls
        }


import os
import torch
from pointnet2 import pointnet2_utils
import numpy as np

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        print(' ==> loading from checkpoint {}.'.format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        print(' ==> load done from epoch {}.'.format(epoch))
    else:
        raise FileNotFoundError

    return epoch


def FPS_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
        sample_indices: np,
    Return:
        sample_indices: sampled pointcloud index, [B, npoint]
    """
    N, _ = xyz.shape
    sample_indices = np.zeros(npoint, dtype=np.int)
    farthest_index = np.random.randint(0, N, dtype=np.int)
    distance = np.ones(N) * 1e10
    for i in range(npoint):
        sample_indices[i] = farthest_index
        centroid = xyz[farthest_index, :]
        dist2 = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist2 < distance
        distance[mask] = dist2[mask]
        farthest_index = np.argmax(distance)
    return sample_indices


def FPS_cuda(points, npoint):
    points_cuda = torch.from_numpy(points).float().cuda()
    points_cuda = points_cuda.unsqueeze(0)
    with torch.no_grad():
        index_cuda = pointnet2_utils.furthest_point_sample(
            points_cuda, npoint)
    return index_cuda.squeeze(0).cpu().numpy()


class Saver():
    def __init__(self, save_dir, max_files=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.log_list = []
        self.save_dir = save_dir
        self.max_files = max_files
        self.saver_log_path = os.path.join(save_dir, '.saver_log')
        if os.path.isfile(self.saver_log_path):
            with open(self.saver_log_path, 'r') as f:
                self.log_list = f.read().splitlines()


    def save_checkpoint(self, model, epoch, name='checkpoint', best=False):
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        if best:
            ckpt_name = '{}.pth'.format(name)
        else:
            ckpt_name = '{}_epoch_{}.pth'.format(name, epoch)
        
        save_path = os.path.join(self.save_dir, ckpt_name)
        state = {'epoch': epoch, 'model_state': model_state}
        torch.save(state, save_path)

        if not best:
            self.log_list.insert(0, save_path)
            if len(self.log_list) > self.max_files:
                pop_file = self.log_list.pop()
                if pop_file != save_path:
                    if os.path.isfile(pop_file):
                        os.remove(pop_file)

            with open(self.saver_log_path, 'w') as f:
                for log in self.log_list:
                    f.write(log + '\n')

import torch
import numpy as np
import argparse
import h5py
import os

from tqdm import tqdm
from knn_cuda import KNN

from utils import FPS_cuda

parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument("--category", type=str, default='Chair')
parser.add_argument("--level", type=int, default=3)
parser.add_argument("--scale", type=float, default=3.5)
parser.add_argument("--sample_nps", type=int, default=1024)
parser.add_argument("--save_dir", type=str, default='./datas/partnet/pre_sampler')
args = parser.parse_args()


def knn_self_cuda(points, k=10):
    '''
    points: numpy (N, 3)
    '''
    knn = KNN(k=k, transpose_mode=True)
    points_cuda = torch.from_numpy(points).float().cuda()
    points_cuda = points_cuda.unsqueeze(0).contiguous()
    _, idx = knn(points_cuda, points_cuda)
    return idx.squeeze(0).data.cpu().numpy()


def detect_edge_points(points, labels, k=10):
    '''
    points: N, 3
    labels: N,
    '''
    idx = knn_self_cuda(points, k)
    kneb_labels = labels[idx]
    kneb_labels = np.sum(kneb_labels, axis=1)
    edge_idx = np.where(labels * k != kneb_labels)[0]
    return edge_idx


def Edge_FPS(points, labels, edge_w=3.5, clip=0.75,
        use_cuda=True, sample_nps=1024):
    
    edge_idx = detect_edge_points(points, labels)
    edge_mask = np.zeros([points.shape[0],])
    edge_mask[edge_idx] = 1

    no_edge_idx = np.where(edge_mask == 0)[0]
    no_edge_points = points[no_edge_idx, :]
    edge_points = points[edge_idx, :]

    spoint = sample_nps
    edge_np = edge_idx.shape[0] / points.shape[0] * edge_w
    edge_np = int(np.ceil(spoint * edge_np))
    edge_np = min(edge_np, int(spoint * clip), edge_idx.shape[0])
    no_edge_np = spoint - edge_np
    
    edge_sidx = FPS_cuda(edge_points, npoint=edge_np)
    edge_sidx = edge_idx[edge_sidx]

    no_edge_sidx = FPS_cuda(no_edge_points, npoint=no_edge_np)
    no_edge_sidx = no_edge_idx[no_edge_sidx]

    return np.concatenate([edge_sidx, no_edge_sidx], axis=0)


if __name__ == '__main__':
    data_folder = './datas/partnet'
    save_folder = args.save_dir
    category = args.category
    level = args.level

    split_list = ['val', 'test', 'train']
    for split in split_list:
        filelist = os.path.join(data_folder,
                        'sem_seg_h5',
                        '{}-{}'.format(category, level), 
                        '{}_files.txt'.format(split)) # <data_folder>/<category>-<level>/<split>_files.txt

        points = []
        labels = []
        folder = os.path.dirname(filelist)
        with open(filelist, 'r') as f:
            for line in f:
                data = h5py.File(os.path.join(folder, line.strip()))
                points.append(data['data'][...].astype(np.float32))
                labels.append(data['label_seg'][...].astype(np.int64))

        points_list = np.concatenate(points, axis=0)
        labels_list = np.concatenate(labels, axis=0)

        edge_fps_indices = []
        for points, labels in tqdm(zip(points_list, labels_list)):
            edge_fps_idx = Edge_FPS(points, labels, edge_w=args.scale, sample_nps=args.sample_nps)

            edge_fps_points = points[edge_fps_idx]
            edge_fps_indices.append(edge_fps_idx)
        
        edge_fps_indices = np.stack(edge_fps_indices, axis=0)
        save_dir = os.path.join(save_folder, '{}-{}'.format(category, level))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, '{}_lv2.npy'.format(split))
        np.save(save_path, edge_fps_indices)
        print(edge_fps_indices.shape, save_path)
            
    with open(os.path.join(save_folder, '{}-{}'.format(category, level), 'args.txt'), 'w') as f:
        print(args, file=f)


import numpy as np
import h5py
import os

from tasks.base.dataset import Dataset_base


def get_dataset(cfgs, split, **kwargs):
    name = cfgs.name
    if name == 'baseline':
        DATASET =  Dataset_baseline
    elif name == 'joint':
        DATASET = Dataset_joint
    else:
        raise NotImplementedError

    if split in ['val', 'eval']:
        split = 'val'
    return DATASET(cfgs, split)


class Dataset_baseline(Dataset_base):
    def load_data(self):
        print(' <dataset> -- load data from {}, {}-{}, {} split.'
            .format(self._cfgs.name, 
                    self._cfgs.category, 
                    self._cfgs.level, 
                    self._split))
        
        # load data
        file_list = os.path.join(self._cfgs.path,
                    'sem_seg_h5',
                    '{}-{}'.format(self._cfgs.category, self._cfgs.level), 
                    '{}_files.txt'.format(self._split))

        class_file_list = os.path.join(self._cfgs.path,
                    'stats/after_merging_label_ids',
                    '{}-level-{}.txt'.format(self._cfgs.category, self._cfgs.level))
        
        with open(class_file_list, 'r') as f:
            self._num_class = len(f.readlines()) + 1 # with other

        points = []
        labels = []
        folder = os.path.dirname(file_list)
        with open(file_list, 'r') as f:
            for line in f:
                data = h5py.File(os.path.join(folder, line.strip()), 'r')
                points.append(data['data'][...].astype(np.float32))
                labels.append(data['label_seg'][...].astype(np.int64))

        self._points = np.concatenate(points, axis=0).tolist()
        self._labels = np.concatenate(labels, axis=0).tolist()

        # label weights
        self._label_weights = np.ones(self._num_class)
        if self._cfgs.use_label_weights and self._split is 'train':
            print(' <dataset> -- set label weights')
            labelweights = np.zeros(self._num_class)
            for seg in self._labels:
                tmp, _ = np.histogram(seg, range(self._num_class + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self._label_weights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        
        # repeat data
        self._total_num_data = len(self._points)
        self._repeat_total_num = self._total_num_data
        if self._split == 'train' and self._total_num_data < self._cfgs.repeat_until_num:
            self._repeat_total_num = (self._cfgs.repeat_until_num // self._total_num_data + 1) * self._total_num_data
    
    def get_data(self, index):
        ret_points = np.array(self._points[index])
        ret_labels = np.array(self._labels[index])
        ret_label_weights = np.array(self._label_weights[ret_labels])
        return {
            'pointcloud': ret_points.astype(np.float32), 
            'seg_labels': ret_labels.astype(np.int64), 
            'label_weights': ret_label_weights.astype(np.float32),
            'num_class': self._num_class
        }


class Dataset_joint(Dataset_baseline):
    def load_data(self):
        super().load_data()

        print(' <dataset> -- load prefps sampler indices from {}.'.format(self._cfgs.prefps_path))
        load_path = os.path.join(self._cfgs.prefps_path,
                '{}-{}/{}_lv2.npy'.format(self._cfgs.category, self._cfgs.level, self._split))
        self._prefps_indices = np.load(load_path)

    def get_data(self, index):
        input_dict = super().get_data(index)
        ret_indices = self._prefps_indices[index]
        ret_prefps_points = input_dict['pointcloud'][ret_indices]
        input_dict['prefps_points'] = ret_prefps_points.astype(np.float32)
        return input_dict


if __name__ == '__main__':
    from utils.config import load_config
    cfgs = load_config('./tasks/partnet_seg/configs/baseline.yaml')
    dst = PartNet_Dataset_PREFPS(cfgs=cfgs.dataset, split='val')
    print(len(dst))
    print(dst.num_class)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dst, batch_size=12, shuffle=False, pin_memory=False, num_workers=0)
    for i, batch in enumerate(loader):
        input_dict = batch
        print(input_dict['pointcloud'].shape)
        print(input_dict['prefps_points'].shape)
        break

from torch.utils.data import Dataset

class Dataset_base(Dataset):
    def __init__(self, cfgs, split):
        super().__init__()
        assert split in ['train', 'val', 'test', 'eval'], 'invalid split: {}'.format(split)
        self._split = split
        self._cfgs = cfgs
        self._total_num_data = None
        self._repeat_total_num = None
        self.load_data()
        
        print(' <dataset> -- load done.')
        print(' <dataset> -- total data:', self._total_num_data)
        print(' <dataset> -- repeat data:', self._repeat_total_num)
    
    def load_data(self):
        raise NotImplementedError

    def get_data(self, index):
        raise NotImplementedError

    def start_iteration(self, epoch=-1):
        return

    def __len__(self):
        return self._repeat_total_num

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration()

        if index >= self._total_num_data:
            index = index % self._total_num_data

        return self.get_data(index)

    @property
    def num_class(self):
        return self._num_class

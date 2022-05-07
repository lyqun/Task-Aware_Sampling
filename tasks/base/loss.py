from torch import nn

class LossFunc_base(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self._cfgs = cfgs

    def update(self, epoch):
        pass

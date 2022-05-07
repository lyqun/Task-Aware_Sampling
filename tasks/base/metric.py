import math

class Metric_base(object):
    def __init__(self, name, is_min=False):
        self._name = name
        self._best_epoch = None
        self._best_value = math.inf
        self._is_min = is_min
        if not self._is_min:
            self._best_value = -self._best_value

    def eval_reset(self):
        raise NotImplementedError

    def eval_bs(self, input_dict, pred_dict):
        raise NotImplementedError

    def eval_detach(self):
        # reset tmp vals
        # return eval val
        raise NotImplementedError

    def update_best(self):
        new_val = self.eval_detach()
        state = False
        if (new_val > self._best_value) ^ self._is_min:
            self._best_value = new_val
            state = True
        return state, new_val

    @property
    def best_val(self):
        return self._best_value

    @property
    def name(self):
        return self._name

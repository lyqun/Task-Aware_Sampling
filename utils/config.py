import yaml
from easydict import EasyDict

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    base_cfg_path = cfg.get('base_cfg')
    if base_cfg_path is not None:
        base_cfg = load_config(base_cfg_path)
    else:
        base_cfg = dict()

    update_recursive(base_cfg, cfg)
    return EasyDict(base_cfg)


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

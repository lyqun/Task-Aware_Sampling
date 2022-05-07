## Task-Aware Sampling Layer for Point-Wise Analysis

Yiqun Lin, Lichang Chen, Haibin Huang, Chongyang Ma, Xiaoguang Han, Shuguang Cui, "Task-Aware Sampling Layer for Point-Wise Analysis", TVCG 2022. [[paper](https://arxiv.org/pdf/2107.04291.pdf)]

### 0. Citation

```
@ARTICLE{lin2022sampling,
  author={Lin, Yiqun and Chen, Lichang and Huang, Haibin and Ma, Chongyang and Han, Xiaoguang and Cui, Shuguang},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Task-Aware Sampling Layer for Point-Wise Analysis}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TVCG.2022.3171794}
}
```

### 1. Environment

This code has been tested with gcc 5.5, Python 3.7, PyTorch 1.8, and CUDA 10.2.

```shell
conda ceate -n env_test python=3.7
source env.sh
pip install torch torchvision
pip install tqdm msgpack six tabulate termcolor pyyaml easydict

# install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# install pointnet2
cd pointnet2
python setup.py install
```

### 2. Data Preparation

Download PartNet semantic segmentation dataset from https://www.shapenet.org/ and unzip them to `./datas/partnet/`. Download the `stats` folder from https://github.com/daerduoCarey/partnet_dataset/tree/master/stats and put it to `./datas/partnet/stats`

Run the following command to generate Edge-FPS sampling points:

```shell
python ./utils/edge_fps.py
```

The folder should be organized as follows:

```shell
./datas/partnet/
├── sem_seg_h5
│   ├── Chair-3
│   │   ├── train_files.txt
│   │   ├── val_files.txt
│   │   ├── *.h5
├── stats
│   ├── after_merging_label_ids
│   │   ├── Chair-level-3.txt
├── pre_sampler
│   ├── Chair-3
│   │   ├── args.txt
│   │   ├── *.npy
```

### 3. Training

Run the following command for training (Chair-3).

```shell
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py \
  --cfg_path ./tasks/partnet_seg/configs/baseline.yaml \
  --save_dir logs/baseline
```

### 4. Testing

Run the following command for testing (Chair-3).

```shell
CUDA_VISIBLE_DEVICES=0 python ./tools/test.py \
  --cfg_path ./tasks/partnet_seg/configs/baseline.yaml \
  --save_dir logs/baseline \
  --resume_metric part_miou
```

| Model          | Config        | Shape mIoU | Part mIoU |
| -------------- | ------------- | ---------- | --------- |
| Baseline (FPS) | baseline.yaml | 49.8       | 40.4      |
| Joint          | joint.yaml    | 51.0       | 41.1      |
| Edge-FPS       | prefps.yaml   | 54.2       | 44.0      |

## License

This repository is released under MIT License (see LICENSE file for details).


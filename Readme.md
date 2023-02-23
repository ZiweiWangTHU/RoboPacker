## Installation
1\) Environment requirements
* Python 3.x
* Pytorch 1.7 or higher
* CUDA 9.2 or higher

Create a conda virtual environment and activate it.
```
conda create -n realsense python=3.7
conda activate realsense
```

Clone and install the following projects
```commandline
https://github.com/facebookresearch/Detic
https://github.com/dbolya/yolact
```

3\) Install the dependencies.
```
pip install matplotlib
pip install numpy
pip install opencv-contrib-python
pip install opencv-python
pip install pybullet
pip install trimesh
pip install pyrealsense2
```
4\) Build CD_loss
```
cd realsense
cd chamer3D
python setup.py install
```
## Prepare Data
1\) Main File Directory
```
realsense
├── urdf_tmp
│   ├── Ball
│   ├── can
│    ...
├── data
│   ├── tmp
├── res
│   ├── 048322070276
│   ├── 048522075245
│    ...
```
```
"urdf_tmp" contains information such as obj model, heightmap, texture, etc. for each class
"data" contains the normalized point cloud template for each class
"res" contains RGB-D camera calibration parameters, and each camera is represented by a serial number
```
## Train
1\) Fine-tuning the pre-trained model
```
CUDA_VISIBLE_DEVICES=0 python train_net.py --num_gpus 1 --config-file configs/Detic_LCOCOI21K_CLIP_SwinB_896B32_4x_ft4x_max-size.yaml
```
2\) Train scale model
```
CUDA_VISIBLE_DEVICES=0 python train_shape_trans.py
CUDA_VISIBLE_DEVICES=0 python train_scale_model.py
```

## Test
```
CUDA_VISIBLE_DEVICES=0,1 python explore_grasp.py
```
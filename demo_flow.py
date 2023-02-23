from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_train_cate, load_pts_tmp_cate, load_pts_tmp_cate_val
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from uti_tool import data_augment
from yolov3_fsnet.detect_fsnet_train import det

from pyTorchChamferDistance.chamfer_distance import ChamferDistance
import chamfer3D.dist_chamfer_3D
from models.models.pointnet2_flow import *
from models.models.loss_helper import *
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models', help='output folder')
parser.add_argument('--outclass', type=int, default=2, help='point class')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()

kc = opt.outclass
num_cor = 3
num_vec = 8
nw = 0  # number of cpu
localtime = (time.localtime(time.time()))
year = localtime.tm_year
month = localtime.tm_mon
day = localtime.tm_mday
hour = localtime.tm_hour

# 1008 = 1158 - 150
model = DeformFlowNet(additional_channel=0, Is_MSG=False)
model.eval()
model.cuda()
Tes = "./out_models/shape_trans_no_msg/Trans_epoch60_objmix_all.pth"
model.load_state_dict(torch.load(Tes))
base_path = os.getcwd()

cats = "can"
tmp_base_path = os.path.join(base_path, "data", "scale_obj_point_train_v3", "tmp")
tmp_path = os.path.join(tmp_base_path, cats + ".txt")
cate_tmp = np.loadtxt(tmp_path)
cate_tmp_print = cate_tmp.copy()

point_base_path = os.path.join(base_path, "input")

pts = glbo.glob("input/*.txt")
pts_list = []

for pts_path in pts:
    pts_one = np.loadtxt(pts_path)
    pts_one = pts_one[:, :3]
    choice = np.random.choice(pts_one.shape[0], 5000, replace=True)
    pts_one = pts_one[choice, :]


    xyz_min = np.min(pts_one[:, 0:3], axis=0)
    xyz_max = np.max(pts_one[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    pts_one[:, 0:3] = pts_one[:, 0:3] - xyz_move
    scale = np.max(np.sqrt(np.sum(pts_one[:, 0:3] ** 2, axis=1)))
    pts_one[:, 0:3] = pts_one[:, 0:3]/scale
    pts_list.append(pts_one.reshape([1, -1, 3]))

pts_list = np.concatenate(pts_list, axis=0)  # [b, n, 3]
pts_all = torch.from_numpy(pts_list)
points = Variable(torch.Tensor(pts_all.float()))
points = points.cuda()
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

cate_tmp = torch.from_numpy(cate_tmp).unsqueeze(0)
cate_tmp = Variable(torch.Tensor(cate_tmp.float()))
cate_tmp = cate_tmp.cuda()

points = points.permute(0, 2, 1)
# target_ = target_.permute(0, 2, 1)
cate_tmp = cate_tmp.permute(0, 2, 1)

for i in range(points.shape[0]):
    points_one = points[i, :, :]
    points_one = points_one.unsqueeze(0)
    with torch.no_grad():
        flow, uncertain_logits = model(cate_tmp, points_one)
    flow = flow.permute(0, 2, 1)  # [b, 3, n] --> [b, n, 3]

    cate_tmp = cate_tmp.permute(0, 2, 1)

    xyz_deform_template = flow + cate_tmp
    xyz_deform_template = xyz_deform_template[0].cpu().detach().numpy()
    print(xyz_deform_template.shape)
    ax = plt.subplot(111, projection='3d')
    base_path_0 = "./Fsnet"
    plt.axis('off')
    ax.grid(False)

    # cate_tmp_print
    ax.scatter(xyz_deform_template[:, 0], xyz_deform_template[:, 1], xyz_deform_template[:, 2], c='g', s=20, marker='.')
    ax.scatter(cate_tmp_print[:, 0], cate_tmp_print[:, 1], cate_tmp_print[:, 2], c='b', s=20, marker='.')

    path = os.path.join(base_path_0, "shape" + str(i) + '.png')
    plt.savefig(path, transparent=True, dpi=800)
    # plt.show()
    plt.close()



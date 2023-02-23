from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_shape, load_pts_shape_val
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import chamfer3D.dist_chamfer_3D
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=700, help='number of epochs to train for')
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

classifier_ce = Point_center_res_cate() ## translation estimation
classifier_ce = nn.DataParallel(classifier_ce)
classifier_ce = classifier_ce.eval()
classifier_ce.cuda()

base_path = os.getcwd()


Tes = "./out_models/scale_branch_lrx1/Classifier_ce_epoch680_objmix_all.pth"
classifier_ce.load_state_dict(torch.load(Tes))


cats = "can"
tmp_base_path = os.path.join(base_path, "data", "scale_pts_train_v1", "tmp")
tmp_path = os.path.join(tmp_base_path, cats + ".txt")
tmp = np.loadtxt(tmp_path)
scale_unit = (np.max(tmp, axis=0) - np.min(tmp, axis=0)) * 1000
print(scale_unit)

scale_pts = np.loadtxt(os.path.join(base_path, "input", "with_scale_tmp", "can_2" + ".txt"))

point_base_path = os.path.join(base_path, "input")

pts = glob.glob("input/*.txt")
pts_list = []
print(pts)
for pts_path in pts:
    pts_one = np.loadtxt(pts_path)
    pts_one = pts_one[:, :3]
    choice = np.random.choice(pts_one.shape[0], 5000, replace=True)
    pts_one = pts_one[choice, :]
    pts_list.append(pts_one.reshape([1, -1, 3]))

pts_list = np.concatenate(pts_list, axis=0)  # [b, n, 3]
pts_all = torch.from_numpy(pts_list)
points = Variable(torch.Tensor(pts_all.float()))
points = points.cuda()
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
cats_list = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
obj_id = np.asarray(cats_list.index(cats) + 1)
obj_id = torch.from_numpy(obj_id).reshape([1])
obj_id = obj_id.cuda()
print(obj_id.shape)
points = points.permute(0, 2, 1)

for i in range(points.shape[0]):
    points_one = points[i, :, :] * 1000.0
    points_one = points_one.unsqueeze(0)
    with torch.no_grad():
        cen_pred, obj_size = classifier_ce((points_one - points_one.mean(dim=2, keepdim=True)), obj_id)

    obj_size = obj_size.cpu().detach().numpy()[0]
    print(obj_size)
    obj_size = obj_size + scale_unit
    scale_factor = obj_size / scale_unit
    cate_tmp_print = tmp.copy()
    cate_tmp_print[:, 0] = cate_tmp_print[:, 0] * scale_factor[0]
    cate_tmp_print[:, 1] = cate_tmp_print[:, 1] * scale_factor[1]
    cate_tmp_print[:, 2] = cate_tmp_print[:, 2] * scale_factor[2]

    xyz_min = np.min(cate_tmp_print[:, 0:3], axis=0)
    xyz_max = np.max(cate_tmp_print[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    cate_tmp_print = cate_tmp_print - xyz_move

    xyz_min = np.min(scale_pts[:, 0:3], axis=0)
    xyz_max = np.max(scale_pts[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    scale_pts = scale_pts - xyz_move

    ax = plt.subplot(111, projection='3d')
    base_path_0 = "./Fsnet"
    plt.axis('off')
    ax.grid(False)

    ax.scatter(scale_pts[:, 0], scale_pts[:, 1], scale_pts[:, 2], c='g', s=20, marker='.')
    ax.scatter(cate_tmp_print[:, 0], cate_tmp_print[:, 1], cate_tmp_print[:, 2], c='b', s=20, marker='.')

    pts = scale_pts.copy().reshape([1, -1, 3])
    new_pts = cate_tmp_print.copy().reshape([1, -1, 3])

    pts = torch.from_numpy(pts).float().cuda()
    new_pts = torch.from_numpy(new_pts).float().cuda()

    dist1, dist2, idx1, idx2 = chamLoss(pts, new_pts)
    loss_1 = torch.mean(dist1) + torch.mean(dist2)

    loss_1 = loss_1.cpu().detach().numpy()


    path = os.path.join(base_path_0, "cd_" + str(loss_1) + '.png')
    plt.savefig(path, transparent=True, dpi=800)
    #plt.show()
    plt.close()











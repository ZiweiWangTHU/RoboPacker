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


def normal_pts(data):
    #centre
    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    #centroid = np.mean(data[:, 0:3], axis=0)
    #data[:, 0:3] = data[:, 0:3] - centroid
    data[:, 0:3] = data[:, 0:3] - xyz_move
    #scale
    #scale = np.max(data[:, 0:3])
    #data[:, 0:3] = data[:, 0:3] / scale
    # scale = np.max(np.linalg.norm(data[:, 0:3], 1))
    scale = np.max(np.sqrt(np.sum(data[:, 0:3] ** 2, axis=1)))
    data[:, 0:3] = data[:, 0:3]/scale
    return data

def shift_pts(data):
    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    # centroid = np.mean(data[:, 0:3], axis=0)
    # data[:, 0:3] = data[:, 0:3] - centroid
    data[:, 0:3] = data[:, 0:3] - xyz_move
    return data

def get_scale(DeformFlowNet, scale_net, tmp_list, pts_list):

    scale_list = []
    scale_pts = []
    scale_uncertainty_list = []
    # target_ = target_.permute(0, 2, 1)

    for points in pts_list:
        cate_tmp = tmp_list[pts_list.index(points)]
        unit_scale = np.max(cate_tmp, axis=0) - np.min(cate_tmp, axis=0)
        points = Variable(torch.Tensor(points.float()))
        points = points.cuda()
        with torch.no_grad():
            cen_pred, obj_size, obj_mu, obj_log_var = scale_net((points - points.mean(dim=2, keepdim=True)), obj_id)
            points = points.permute(0, 2, 1)
            cate_tmp = cate_tmp.permute(0, 2, 1)
            flow, uncertain_logits = DeformFlowNet(cate_tmp, points)
            flow = flow.permute(0, 2, 1)  # [b, 3, n] --> [b, n, 3]
            cate_tmp = cate_tmp.permute(0, 2, 1)
            xyz_deform_template = flow + cate_tmp
            xyz_deform_template = xyz_deform_template[0].cpu().detach().numpy()

        scale_factor = obj_size / unit_scale
        std = torch.exp(0.5 * obj_log_var)
        eps = torch.randn_like(std)
        scale_uncertainty = eps * std + obj_mu
        xyz_deform_template[:, 0] = xyz_deform_template[:, 0] * scale_factor[0]
        xyz_deform_template[:, 1] = xyz_deform_template[:, 1] * scale_factor[1]
        xyz_deform_template[:, 2] = xyz_deform_template[:, 2] * scale_factor[2]
        scale_uncertainty_list.append(scale_uncertainty)
        scale_pts.append(xyz_deform_template)
        scale_list.append(scale_factor)
    return scale_list, scale_pts, scale_uncertainty_list
import argparse
import pickle
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov3_fsnet.utils.datasets import LoadImages_fsnet
from yolov3_fsnet.utils.general import check_img_size

from uti_tool import getFiles_ab_cate, depth_2_mesh_bbx, load_ply, show_3D_single, loss_recon
from Net_deploy import load_models, FS_Net_Test
from torch.utils.data import DataLoader
from yolact_fsnet.yolact_infer import yolact_main
from yolact_fsnet.layers.box_utils import mask_iou, jaccard

K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

def chooselimt_test(pts0, dia, cen):  ##replace the 3D sphere with 3D cube

    pts = pts0.copy()
    pts = pts[np.where(pts[:, 2] > 20)[0], :]
    ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia)[0], :]
    if ptsn.shape[0] < 1000:
        ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 2)[0], :]
        if ptsn.shape[0] < 500:
            ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 3)[0], :]
    return ptsn

def pose_estimate(rgb,depth,label):

    bbbox = label['bboxes']
    cate = label['class_ids']
    Rt = gts['rotations'].reshape(3, 3)
    Tt = gts['translations'].reshape(1, 3) * 1000.0

    model = load_ply('/home/lcl/fsnet_oper/yolov3_fsnet/trained_models/' + cate + '.ply')  # load model point cloud
    pc = model['pts'] * 1000.0

    classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0 = load_models(
        cate)

    dep3d = depth_2_mesh_bbx(depth, [bbbox[0], bbbox[2], bbbox[1], bbbox[3]], K)

    dep3d = dep3d[np.where(dep3d[:, 2] > 0.0)]

    cen_depth = np.array([-100, -180, 500]).reshape((1, 3))
    dep3d = chooselimt_test(dep3d, 400, cen_depth)
    choice = np.random.choice(len(dep3d), 2000, replace=True)
    dep3d = dep3d[choice, :]
    show_3D_single(dep3d, 'ori')

    iou3d, R_loss, T_loss, Rec_loss, Seg_loss, pts_s1, rec_seg_loss, loss_T = FS_Net_Test(dep3d, pc, rgb,
                                                                                  Rt, Tt, classifier_seg3D,
                                                                                  classifier_ce,
                                                                                  classifier_Rot_green,
                                                                                  classifier_Rot_red,
                                                                                  cate, model_size,
                                                                                  cate_id0, seg, pts_seg,cen_gt,
                                                                                  num_cor=3,
                                                                                  )

if __name__ == '__main__':
    path = '/home/lcl/display_5/'
    cate = 'cracker_box'
    rgb = cv2.imread(path + cate +'_1_rgb.png')
    depth = cv2.imread(path + cate +'_1_depth.png', -1)
    gts = pickle.load(open(path + cate +'_1_label.pkl', 'rb'))
    R, T, size = pose_estimate(rgb, depth, gts)
import pickle
import torch

import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from yolov3_fsnet.utils.datasets import LoadImages_fsnet
from yolov3_fsnet.utils.general import check_img_size
from Augment.augmentor import Augmentor
from uti_tool import data_augment
from uti_tool import getFiles_ab_cate, depth_2_mesh_bbx, rotMat_2_Euler

augmentor = Augmentor()
augmentor = nn.DataParallel(augmentor)
augmentor = augmentor.eval()
augmentor.cuda()

aug_model = '/home/lcl/fsnet_oper/models/FS_Net_power_drill_aug/augmentor_last_objpd_aug.pth'
augmentor.load_state_dict(torch.load(aug_model))

# load dataset
cate = 'power_drill'
data_path = '/home/lcl/FS_Net-main/data/%s/' % cate
stride = 32
imgsz = 320
imgsz = check_img_size(imgsz, s=stride)  # check img_size
dataset = LoadImages_fsnet(data_path, img_size=imgsz, stride=stride)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
for icc, data in enumerate(dataloader):
    print(icc)
    path, img, im0s, depth_, Rt, Tt, gt_seg, gt_box, pc, pts_rec, seg, pts_seg, cen_gt = data
    name = data[0][0][:len(data[0][0]) - 7] + 'label.pkl'
    ground = pickle.load(open(name, 'rb'))
    bbbox = ground['bboxes']
    depth = depth_[0].numpy()
    dep3d = depth_2_mesh_bbx(depth, [bbbox[0], bbbox[2], bbbox[1], bbbox[3]], K)
    dep3d = dep3d[np.where(dep3d[:, 2] > 0.0)]
    cen_depth = np.array([-100, -180, 500]).reshape((1, 3))
    choice = np.random.choice(len(dep3d), 2000, replace=True)
    dep3d = dep3d[choice, :]

    ptscp = torch.tensor(dep3d).cuda().float()
    ptscp = ptscp.unsqueeze(0)
    ptscp = ptscp.transpose(2, 1).contiguous()

    noise = 0.02 * torch.randn(1, 1024).cuda()
    pts_aug, aug_for_rot, feat = augmentor(ptscp, noise, 0)
    feat_np = feat.detach().cpu().numpy()
    print('feat:', feat_np[0, 0:20])
    if icc>1:
        delta = abs(feat_np[0]-temp)
        print('delta:',delta)
        print('delta sum:',np.sum(delta))
    temp = feat_np[0]

    pts_aug = pts_aug.transpose(1, 2).contiguous()
    aug_for_rot = aug_for_rot.cpu().detach().numpy()
    points_aug1 = pts_aug.cpu().numpy().copy()
    R0 = rotMat_2_Euler(Rt[0].cpu().numpy().copy())
    R1 = rotMat_2_Euler(aug_for_rot[0])/180*15
    print('angle',R0,R1)
    print()

    noise = 0.1 * torch.randn(1, 1024).cuda()
    pts_aug, aug_for_rot, feat = augmentor(ptscp, noise, 0)
    feat_np = feat.detach().cpu().numpy()
    print('feat:', feat_np[0, 0:20])
    if icc > 1:
        delta = abs(feat_np[0] - temp)
        print('delta:', delta)
        print('delta sum:', np.sum(delta))
    temp = feat_np[0]

    pts_aug = pts_aug.transpose(1, 2).contiguous()
    aug_for_rot = aug_for_rot.cpu().detach().numpy()
    points_aug1 = pts_aug.cpu().numpy().copy()
    R0 = rotMat_2_Euler(Rt[0].cpu().numpy().copy())
    R1 = rotMat_2_Euler(aug_for_rot[0]) / 180 * 15
    print('angle', R0, R1)
    print()


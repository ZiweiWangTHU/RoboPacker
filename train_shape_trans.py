from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable

import torch
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_train_cate, load_pts_tmp_cate, load_pts_tmp_cate_val
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from uti_tool import data_augment

import chamfer3D.dist_chamfer_3D
from models.models.pointnet2_flow import *
from models.models.loss_helper import *

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
model.train()
model.cuda()

cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]

for cat in [["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]]:

    if len(cat) > 1:
        opt.outf = 'out_models/shape_trans'
    else:
        opt.outf = 'out_models/FS_Net_%s' % (cat[0])
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    cat1 = 'mix_all'  # weight file name
    sepoch = 0
    # batch_size = 10  # bathcsize
    batch_size = opt.batchSize
    lr = 0.001
    epochs = 500

    optimizer = optim.Adam([{'params': model.parameters()}],
        lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data", "scale_pts_train_v1")
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    dataloader = load_pts_tmp_cate(data_path, batch_size, cat, shuf=True, drop=True, nw=nw)

    log = open('%s/log.txt' % opt.outf, 'w')

    for epoch in range(sepoch, epochs):
        lr = max(0.001*(0.5**(epoch//20)), 1e-5)
        optimizer.param_groups[0]['lr'] = lr

        for i, data in enumerate(dataloader):


            # points, target_, obj_id = data['points'], data['label'], data['cate_id']
            points, target_, obj_id, cate_tmp = data['points'], data['label'], data['cate_id'], data['model']
            points = Variable(torch.Tensor(points.float()))
            target_ = Variable(torch.Tensor(target_.float()))
            cate_tmp = Variable(torch.Tensor(cate_tmp.float()))
            points, target_ = points.cuda(), target_.cuda()
            cate_tmp = cate_tmp.cuda()

            # [b, n, 3] --> [b, 3, n]
            points = points.permute(0, 2, 1)
            # target_ = target_.permute(0, 2, 1)
            cate_tmp = cate_tmp.permute(0, 2, 1)

            optimizer.zero_grad()

            # xyz_deform_template, flow = model(target_, points)
            flow, uncertain_logits = model(cate_tmp, points)

            flow = flow.permute(0, 2, 1) # [b, 3, n] --> [b, n, 3]
            # target_ = target_.permute(0, 2, 1)
            cate_tmp = cate_tmp.permute(0, 2, 1)

            xyz_deform_template = flow + cate_tmp
            dist1, dist2, idx1, idx2 = chamLoss(xyz_deform_template, target_)
            loss_1 = torch.mean(dist1) + torch.mean(dist2)
            loss_2 = flow_reguliarzer(flow)
            print(uncertain_logits.shape)

            loss_3 = uncertain_loss(uncertain_logits, dist1.mean(-1) + dist2.mean(-1), threshold=0.0003)

            loss = loss_1 + 0.01 * loss_2 + loss_3
            #loss_2 = flow_reguliarzer(flow)

            # loss = loss_1 + loss_2
            # print(loss.shape)
            # print(loss_1.shape)
            # print(loss_2.shape)
            loss.backward()
            optimizer.step()

            print(cats[obj_id[0] - 1])
            log.write(cats[obj_id[0] - 1] + '\n')

            print('[%d: %d] train loss: %f' % (epoch, i, loss.item()))
            log.write('[%d: %d] train loss_seg: %f' % (epoch, i, loss.item()))

            print()

        if epoch >= 0 and epoch % 20 == 0:  ##save mid checkpoints
            eval_log = open('%s/eval_log.txt' % opt.outf, 'a')
            torch.save(model.state_dict(), '%s/Trans_epoch%d_obj%s.pth' % (opt.outf, epoch, cat1))
            model.eval()
            data_val_path = os.path.join(base_path, "data", "scale_obj_point_val")
            cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
            cd_dis_all = []
            cd_dis_one = []
            for cat in tqdm(cats):
                dataloader_val = load_pts_tmp_cate_val(data_val_path, 10, cat, shuf=False, drop=False, nw=0)
                for i, data in enumerate(dataloader_val):
                    points, target_, obj_id, cate_tmp = data['points'], data['label'], data['cate_id'], data['model']
                    points = Variable(torch.Tensor(points.float()))
                    target_ = Variable(torch.Tensor(target_.float()))
                    cate_tmp = Variable(torch.Tensor(cate_tmp.float()))
                    points, target_ = points.cuda(), target_.cuda()
                    cate_tmp = cate_tmp.cuda()
                    points = points.permute(0, 2, 1)
                    cate_tmp = cate_tmp.permute(0, 2, 1)
                    with torch.no_grad():
                        flow, uncertain_logits = model(cate_tmp, points)
                        flow = flow.permute(0, 2, 1)  # [b, 3, n] --> [b, n, 3]
                        cate_tmp = cate_tmp.permute(0, 2, 1)
                        xyz_deform_template = flow + cate_tmp
                        dist1, dist2, idx1, idx2 = chamLoss(xyz_deform_template, target_)
                        loss_1 = torch.mean(dist1) + torch.mean(dist2)
                    cd_dis_one.append(loss_1.cpu().numpy().item())
                cd_dis_all.append(np.mean(np.asarray(cd_dis_one)))
            cd_dis_all = np.asarray(cd_dis_all)
            eval_log.write(str(epoch) + '\n')
            eval_log.write('[%d: %d] train cd_dis: %f' % (epoch, i, np.mean(cd_dis_all).item()))
            eval_log.close()
            model.train()
    log.close()


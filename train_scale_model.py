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

# 加载模型与损失函数
classifier_ce = Point_center_res_cate()
Loss_func_s = nn.MSELoss()
Loss_func_ce = nn.MSELoss()
classifier_ce = nn.DataParallel(classifier_ce)
classifier_ce = classifier_ce.train()
Loss_func_s.cuda()
Loss_func_ce.cuda()
classifier_ce.cuda()

cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]

for cat in [["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]]:

    if len(cat) > 1:
        opt.outf = 'out_models/scale_branch_lrx10'
    else:
        opt.outf = 'out_models/FS_Net_%s' % (cat[0])
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    cat1 = 'mix_all'  # weight file name
    sepoch = 0
    batch_size = opt.batchSize #
    lr = 0.001
    epochs = opt.nepoch

    optimizer = optim.Adam([{'params': classifier_ce.parameters()}],
        lr=lr, betas=(0.9, 0.99))

    base_path = os.getcwd()
    # scale_pts_train_v1
    data_path = os.path.join(base_path, "data", "scale_pts_train_v1")

    K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    dataloader = load_pts_shape(data_path, batch_size, cat, shuf=True, drop=True, nw=nw)

    log = open('%s/log.txt' % opt.outf, 'w')

    for epoch in range(sepoch, epochs):

        if epoch > 0 and epoch % (epochs // 5) == 0:
            lr = lr / 4

        optimizer.param_groups[0]['lr'] = lr * 10

        for i, data in enumerate(dataloader):

            points, label, obj_id, centers = data['points'], data['label'], data['cate_id'], data['model']
            points = Variable(torch.Tensor(points.float()))

            points = points.cuda()
            label = label.cuda()
            centers = centers.cuda()
            # [b, 3, n]
            points = points.permute(0, 2, 1)
            optimizer.zero_grad()
            cen_pred, obj_size, obj_mu, obj_log_var = classifier_ce((points - points.mean(dim=2, keepdim=True)), obj_id)

            # reparameterize
            std = torch.exp(0.5 * obj_log_var)
            eps = torch.randn_like(std)
            sclae_uncertainty = eps * std + obj_mu

            loss_size = Loss_func_s(obj_size, label.float())
            loss_1 = Loss_func_s(sclae_uncertainty, label.float())
            loss_res = Loss_func_ce(cen_pred, centers.float())
            Loss = loss_res/20.0 + loss_size / 20.0 + loss_1 / 20.0
            Loss.backward()
            optimizer.step()

            print(cats[obj_id[0] - 1])
            log.write(cats[obj_id[0] - 1] + '\n')

            print('[%d: %d] loss_size: %f, loss_res: %f, loss_uncertainty: %f' % (epoch, i, loss_size.item(), loss_res.item(), loss_1.item()))
            log.write('[%d: %d] train loss_seg: %f, loss_res: %f, loss_uncertainty: %f' % (epoch, i, loss_size.item(), loss_res.item(), loss_1.item()))

            print()

        if epoch > 0 and epoch % 20 == 0:  ##save mid checkpoints
            eval_log = open('%s/eval_log.txt' % opt.outf, 'a')
            torch.save(classifier_ce.state_dict(), '%s/Classifier_ce_epoch%d_obj%s.pth' % (opt.outf, epoch, cat1))
            classifier_ce.eval()
            data_val_path = os.path.join(base_path, "data", "scale_pts_val_v1")
            cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
            cd_dis_all = []
            cd_dis_one = []
            for cat in tqdm(cats):
                dataloader_val = load_pts_shape_val(data_val_path, 10, cat, shuf=False, drop=False, nw=0)
                for i, data in enumerate(dataloader):
                    points, label, obj_id, cate_tmp = data['points'], data['label'], data['cate_id'], data['model']
                    points = Variable(torch.Tensor(points.float()))
                    points = points.cuda()
                    label = label.cuda()
                    # [b, 3, n]
                    points = points.permute(0, 2, 1)
                    with torch.no_grad():
                        cen_pred, obj_size, _, _ = classifier_ce((points - points.mean(dim=2, keepdim=True)), obj_id)
                        loss_size = Loss_func_s(obj_size, label.float())
                        Loss = loss_size / 20.0
                    cd_dis_one.append(Loss.cpu().numpy().item())
                cd_dis_all.append(np.mean(np.asarray(cd_dis_one)))
            cd_dis_all = np.asarray(cd_dis_all)
            eval_log.write(str(epoch) + '\n')
            eval_log.write('[%d: %d] train cd_dis: %f' % (epoch, i, np.mean(cd_dis_all).item()))
            eval_log.close()
        classifier_ce.train()

    log.close()











# @Time    : 11/05/2021
# @Author  : Wei Chen
# @Project : Pycharm

from __future__ import print_function

import os
from uti_tool import compute_3d_IoU, loss_recon , show_3D_single
import argparse
import numpy as np
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
import torch
import torch.nn as nn
import cv2
import pdb

from uti_tool import load_ply, draw_cors_withsize, draw_cors, get_3D_corner, trans_3d, gettrans, get6dpose1
Loss_func_ce = nn.MSELoss()

def load_models(cat, cat1='pd', outf='1', train=False):
    classifier_seg3D = GCN3D_segR(class_num=2, vec_num=1, support_num=7, neighbor_num=10)
    classifier_ce = Point_center_res_cate()  ## translation estimation
    classifier_Rot_red = Rot_red(F=1296, k=6)  ## rotation red
    classifier_Rot_green = Rot_green(F=1296, k=6)  ### rotation green

    # optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    classifier_seg3D = nn.DataParallel(classifier_seg3D)
    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_Rot_red = nn.DataParallel(classifier_Rot_red)
    classifier_Rot_green = nn.DataParallel(classifier_Rot_green)

    classifier_seg3D = classifier_seg3D.eval()
    classifier_ce = classifier_ce.eval()
    classifier_Rot_red = classifier_Rot_red.eval()
    classifier_Rot_green = classifier_Rot_green.eval()
    #

    classifier_seg3D.cuda()
    classifier_ce.cuda()
    classifier_Rot_green.cuda()
    classifier_Rot_red.cuda()

    if train:
        outf = '/home/lcl/位姿估计/fsnet_oper/%s/' % outf     # trained models
        temp =cat
        cat=cat1
        Seg3d = '%s/Seg3D_last_obj%s.pth' % (outf, cat)
        # cat = temp
        Rot = '%s/Rot_g_last_obj%s.pth' % (outf, cat)
        Rot_res = '%s/Rot_r_last_obj%s.pth' % (outf, cat)
        # cat = 'cb'
        Tes = '%s/Tres_last_obj%s.pth' % (outf, cat)
    else:
        outf = '/home/lcl/位姿估计/fsnet_oper/yolov3_fsnet/trained_models'
        temp =cat
        cat='3mix_model'
        # cat = 'mix_all_ep200'
        Seg3d = '%s/Seg3D_last_obj%s.pth' % (outf, cat)
        # cat = temp
        Rot = '%s/Rot_g_last_obj%s.pth' % (outf, cat)
        Rot_res = '%s/Rot_r_last_obj%s.pth' % (outf, cat)
        # cat = 'cb'
        Tes = '%s/Tres_last_obj%s.pth' % (outf, cat)

    cat =temp
    classifier_seg3D.load_state_dict(torch.load(Seg3d))
    classifier_ce.load_state_dict(torch.load(Tes))
    classifier_Rot_green.load_state_dict(torch.load(Rot))
    classifier_Rot_red.load_state_dict(torch.load(Rot_res))
    # model_sizes = np.array(
    #     [[120,171,39], [138,129,39], [82,48,67], [75,72,121], [346, 200, 335], [93,74,65],    # from grasp
    #      [27, 62, 81], [48, 33, 96], [92, 94, 29], [93, 74, 65], [25, 47, 88]])  ## 6x3
    model_sizes = np.array(
        [[120, 171, 39], [138, 129, 39], [82, 48, 67], [60, 58, 97], [346, 200, 335], [93, 74, 65],
         [35, 82, 107], [48, 33, 96], [92, 94, 29], [93, 74, 65], [25, 47, 88]])  ## 6x3

    cats = ['large_clamp', 'pudding_box', 'potted_meat_can', 'pitcher_base', 'laptop', 'mug',
            'cracker_box', 'mustard_bottle', 'power_drill', 'mug_deformed','sugar_box']
    cate_id0 = np.where(np.array(cats) == cat)[0][0]
    model_size = model_sizes[cate_id0]

    return classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0


def FS_Net_Test(points, pc, rgb, Rt, Tt, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red,
                cat, model_size, cate_id0, seg, pts_seg, cen_gt, num_cor=3, pts_rec=0):
    OR, x_r, y_r, z_r = get_3D_corner(pc)

    points = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)

    Rt0 = Rt[0].numpy()
    Tt = Tt[0].numpy().reshape(3, 1)

    ptsori = points.clone()
    points = points.numpy().copy()  # [b, n, 3] --> [1, n, 3]

    res = np.mean(points[0], 0)
    points[0, :, 0:3] = points[0, :, 0:3] - np.array([res[0], res[1], res[2]])

    points = torch.from_numpy(points).cuda()

    pointsf = points[:, :, 0:3].unsqueeze(2)  ##128 1500 1 12

    points = pointsf.transpose(3, 1)
    points_n = pointsf.squeeze(2)


    cate_id= torch.zeros((1,1))               # add one hot to GCN
    cate_id[0][0] = cate_id0+1
    one_hot = torch.zeros(points.shape[0], 16).scatter_(1, cate_id.cpu().long(), 1)
    one_hot = one_hot.cuda()

    # obj_idh = torch.zeros((1, 1))
    #
    # if obj_idh.shape[0] == 1:
    #     obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    # else:
    #     obj_idh = obj_idh.view(-1, 1)

    # one_hot = torch.zeros(points.shape[0], 16).scatter_(1, obj_idh.cpu().long(), 1)
    #
    # one_hot = one_hot.cuda()
    model = pc[np.random.choice(len(pc), 500, replace=True), :]
    model = torch.from_numpy(model).unsqueeze(0).cuda()
    points_n1 = torch.cat([points_n, model], dim=1)


    pred_seg, point_recon, feavecs = classifier_seg3D(points_n1, one_hot)                    # GCN deploy



    pts_rec = pts_rec.cuda()

    pts_reccp = point_recon

    pts_reccp = pts_reccp.cpu()

    Ttt = Tt.reshape(1, 3)
    pts_recon = pts_reccp[0].numpy()+Ttt

    show_3D_single(pts_recon, 'recon')


    loss_vec = loss_recon(point_recon, pts_rec)
    # print('        %f' % loss_vec.item())

    pred_choice = pred_seg.data.max(2)[1]

    Loss_seg3D = nn.CrossEntropyLoss()
    # Loss_seg3D.cuda()
    loss_seg = Loss_seg3D(pred_seg.reshape(-1, pred_seg.size(-1)).cpu(), seg.view(-1, ).long().cpu())
    p = pred_choice

    ptsori = ptsori.cuda()
    pts_ = torch.index_select(ptsori[0, :, 0:3], 0, p[0, :].nonzero()[:, 0])  ##Nx3     z=500


    # seg points from pred recon and visualize
    rec_seg = torch.index_select(point_recon[0, :, 0:3], 0, p[0, :].nonzero()[:, 0])
    choice_seg = np.random.choice(len(pts_rec), len(rec_seg), replace=True)
    pts_rec_seg = pts_rec[choice_seg, :].unsqueeze(0)
    reg_seg1 = rec_seg.unsqueeze(0)
    loss_vec1 = loss_recon(reg_seg1, pts_rec)
    pts_seg_recon = rec_seg.cpu().numpy() + Ttt
    show_3D_single(pts_seg_recon, 'recon_seg')

    # print('loss rec:%f, loss rec seg:%f' %(loss_vec.item(), loss_vec1.item()))
    # choice = np.random.choice(len(dep3d_seg), len(pts_), replace=True)
    # dep3d_seg = dep3d_seg[choice, :].astype(np.float32)
    # dep3d_seg = torch.tensor(dep3d_seg).unsqueeze(0).cuda()

    # pts_1 = pts_.unsqueeze(0).cuda()
    # loss_seg = loss_recon(pts_1, dep3d_seg)

    feat = torch.index_select(feavecs[0, 0:2000, :], 0, p[0, :].nonzero()[:, 0])
    choice1 = np.random.choice(len(feat), 800, replace=True)
    choice2 = np.random.choice(500, 200, replace=True)
    feat_ori = feat[choice1, :]
    feat_model = feavecs[0, 2000:2500, :][choice2, :]
    feat = torch.cat([feat_ori, feat_model], dim=0)

    if len(pts_) < 10:
        print('No object pts')
    else:
        pts_s = pts_[:, :].unsqueeze(0).float()
        # print(ib)

        # p[0, 10:31]
        # feas = torch.index_select(feass[ib, :, :], 0, indexs[ib, :].nonzero()[:, 0])

        if num_cor == 3:
            corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]]))
        else:
            corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0]]))

        pts_bak = pts_s.clone()
        pts_s = pts_s.cuda()
        feat = feat.cuda()
        corners0 = corners0.cuda()



        pts_s1 = np.asarray(pts_bak[0].cpu())
        show_3D_single(pts_s1, 'seg')
        # choice = np.random.choice(1500, len(pts_), replace=True)
        # pts_seg = pts_seg[:,choice, :]
        # pts_s1 = torch.tensor(pts_s1).unsqueeze(0).cuda()
        # loss_seg=loss_recon(pts_s1,pts_seg)

        pts_s = pts_s.transpose(2, 1)



        # pts_s = point_recon.transpose(2, 1)

        cate_id0 +=1

        cen_pred, obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), torch.Tensor([cate_id0]))
        # print(cen_pred)
        pts_seg_recon =pts_seg_recon.transpose(1,0)

        # cen_gt = Tt-np.mean(pts_seg_recon,1).reshape(3,1)
        cen_gt = cen_gt[0].cuda()
        loss_T=Loss_func_ce(cen_gt,cen_pred)
        # print('loss_T:',loss_T.item())
        # cen_pred, obj_size = classifier_ce(point_recon.transpose(2, 1), torch.Tensor([cate_id0]))
        T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)  ## 1x3x1

        # feavec = torch.cat([box_pred, feat.unsqueeze(0)], 2)  ##
        feavec = feat.unsqueeze(0).transpose(1, 2)
        kp_m = classifier_Rot_green(feavec)

        if num_cor == 3:
            corners_ = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        else:
            corners_ = np.array([[0, 0, 0], [0, 1, 0]])

        kpm_gt = (trans_3d(corners_, Rt0, np.array([0, 0, 0]).T).T).flatten()

        bbx_3D = model_size + obj_size.detach().cpu().numpy()
        model_3D = np.array([x_r, y_r, z_r])

        box_pred_gan = classifier_Rot_red(feat.unsqueeze(0).transpose(1, 2))

        pred_axis = np.zeros((num_cor, 3))

        pred_axis[0:2, :] = kp_m.view((2, 3)).detach().cpu().numpy()
        if num_cor == 3:
            pred_axis[2, :] = box_pred_gan.view((2, 3)).detach().cpu().numpy()[1, :]

        box_pred_gan = box_pred_gan.detach().cpu().numpy()
        box_pred_gan = box_pred_gan / np.linalg.norm(box_pred_gan)
        cor0 = corners0.cpu().numpy()
        cor0 = cor0 / np.linalg.norm(cor0)
        kpm_gt = kpm_gt.reshape((num_cor, 3))
        kpm_gt = kpm_gt / np.linalg.norm(kpm_gt)

        pred_axis = pred_axis / np.linalg.norm(pred_axis)

        pose_gt = gettrans(cor0.reshape((num_cor, 3)), kpm_gt.reshape((num_cor, 1, 3)))
        Rt = pose_gt[0][0:3, 0:3]

        pose = gettrans(cor0.reshape((num_cor, 3)), pred_axis.reshape((num_cor, 1, 3)))
        R = pose[0][0:3, 0:3]

        T = (pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)).view(1, 3).detach().cpu().numpy()



        # T = res[0:3]+( cen_pred.unsqueeze(2)).view(1, 3).detach().cpu().numpy()
        # noise_batch_drop_numofloss_loss__cls_model_epoch.pth
        torch.cuda.empty_cache()

        show = 1
        if show == 1:
            R_loss, T_loss = get6dpose1(Rt, Tt, R, T, cat)

            # print(Tt,T)
            size_2 = bbx_3D.reshape(3)
            K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

            rgb0 = rgb
            rgb0 = draw_cors(rgb0, pc, K, Rt, Tt, [0, 255, 0])
            rgb0 = draw_cors_withsize(rgb0, K, R, T, [255, 0, 0], xr=size_2[0], yr=size_2[1], zr=size_2[2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb0, 'R_loss: %s' % (R_loss), (10, 20), font, 0.5, (0, 0, 0), 1, 0)
            cv2.putText(rgb0, 'T_loss(mm): %s' % (T_loss), (10, 40), font, 0.5, (0, 0, 0), 1, 0)
            cv2.imshow('show', rgb0 / 255)
            cv2.waitKey(10)
        eva = 1
        if eva == 1:
            sRT_1 = np.eye(4)
            sRT_1[0:3, 0:3] = Rt
            sRT_1[0:3, 3:4] = Tt
            sRT_2 = np.eye(4)
            sRT_2[0:3, 0:3] = R
            sRT_2[0:3, 3:4] = T.reshape(3, 1)
            size_2 = bbx_3D.reshape(3)
            size_1 = model_3D

            # size_2 = size_1
            class_name_1 = cat
            class_name_2 = cat
            iou3d = compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2,
                                   handle_visibility=1)

            ####print(110, iou3d, R_loss, T_loss, loss_vec.item(), loss_seg.item(), pts_s1, loss_vec1.item(),loss_T.item())
            return iou3d, R_loss, T_loss, loss_vec.item(), loss_seg.item(), pts_s1, loss_vec1.item(),loss_T.item()

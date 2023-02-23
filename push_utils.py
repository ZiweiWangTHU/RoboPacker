import numpy as np
from layers.output_utils import postprocess, undo_image_transformation
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from data import cfg, set_cfg, set_dataset
from yolact import Yolact
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time

from my_utils.my_utils import *

import matplotlib.pyplot as plt
import cv2


import pyrealsense2 as rs
import socket
from my_utils.ur import get_state, gripper_state, grasp_safe
from my_utils.rs import get_current_image
from my_utils.rigid_transform import pix2wld
from robot import Robot

def push_explorer(robot, net, device, robot_workspace_limits, pipeline_list, e2h_R_cam2base_list, e2h_t_cam2base_list, e2h_mtx_list, e1h_mtx, chamLoss, R_cam2gripper=None, t_cam2gripper=None, use_e1h_camera=True):


    heightmap_resolution = 0.002
    align = rs.align(rs.stream.color)  #
    shift = 0.16
    start_list = []
    ur5_ip = ('192.168.3.60', 30003)
    init_pos = np.asarray([0.350, 0.0, 0.40, 2.222, -2.222, 0.0])


    for test_time in range(4):
        color_img_list, depth_img_list, aligned_depth_frame_list = get_current_images(pipeline_list, align)
        # R_gripper2base, t_gripper2base = gripper_state(ur5_ip)
        R_gripper2base, t_gripper2base = robot.gripper_state()
        view_result_list = []
        segment_heat_map_list = []
        position_list = []
        depth_heightmap_list = []
        mask_list = []
        class_list = []
        score_list = []

        for view_id in range(len(color_img_list)):
            color_img = color_img_list[view_id]
            depth_img = depth_img_list[view_id]
            aligned_depth_frame = aligned_depth_frame_list[view_id]
            frame = torch.from_numpy(color_img).to(device).float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            with torch.no_grad():
                preds = net(batch)
                # 获得image的大小
                h, w, _ = frame.shape
                t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.9)
            view_result_list.append(t)
            segment_map = t[4].permute(1, 2, 0).clone()
            segment_heat_map = entropy_map(segment_map)
            segment_heat_map_list.append(segment_heat_map)

            if use_e1h_camera and view_id == len(color_img_list) - 1:
                position, depth_heightmap = get_point_e1h(depth_img, aligned_depth_frame, e1h_mtx,
                                                           R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
            else:
                position, depth_heightmap = get_points(depth_img, aligned_depth_frame, e2h_mtx_list[view_id],
                                                       e2h_R_cam2base_list[view_id], e2h_t_cam2base_list[view_id])
            position_list.append(position)
            depth_heightmap_list.append(depth_heightmap)

            mask = t[3].clone()
            mask_list.append(mask.cpu().numpy())
            class_one = t[0].clone()
            class_list.append(class_one.cpu().numpy())
            score = t[1].clone()
            score_list.append(score.cpu().numpy())

        if test_time > 1:
            shift = get_shift(start_list, shift=shift)

        mask_sum_list = []
        entropy_one_normal_list = []
        for mask_id in range(len(mask_list)):
            mask_one = mask_list[mask_id]
            mask_one = np.sum(mask_one, axis=0)
            mask_sum_list.append(mask_one)
            # 顺便对entropy_map做归一化
            entropy_one = segment_heat_map_list[mask_id]
            entropy_normal_one = data_normal_2d(entropy_one)
            entropy_one_normal_list.append(entropy_normal_one.cpu())

        # 将各个视角下的熵值投射到世界坐标系
        # 看来heightmap采用点云的方式更加稳妥
        entropy_heightmap_list = []
        mask_heightmap_list = []
        for view_id in range(len(mask_sum_list)):
            position = position_list[view_id]
            entropy_map_all = entropy_one_normal_list[view_id]
            masks = mask_sum_list[view_id]

            position_path = "position" + "_" + str(view_id) + ".npy"
            entropy_map_all_path = "entropy_map_all_path" + "_" + str(view_id) + ".npy"
            masks_path = "masks" + "_" + str(view_id) + ".npy"
            # np.save(position_path, position)
            # np.save(entropy_map_all_path, entropy_map_all)
            # np.save(masks_path, masks)

            # 我们依旧选用top_down视角的深度图
            if view_id == len(mask_sum_list) - 1 and use_e1h_camera:
                _, entropy_heightmap, mask_heightmap = get_heightmap(position[:,
                                                                     :3], robot_workspace_limits,
                    entropy_map_all, masks,
                    heightmap_resolution=heightmap_resolution)
            elif view_id == 0:
                depth_heightmap, entropy_heightmap, mask_heightmap = get_heightmap(position[:,
                                                                                   :3], robot_workspace_limits,
                    entropy_map_all, masks,
                    heightmap_resolution=heightmap_resolution)
            else:
                _, entropy_heightmap, mask_heightmap = get_heightmap(position[:,
                                                                     :3], robot_workspace_limits, entropy_map_all, masks,
                    heightmap_resolution=heightmap_resolution)

            entropy_heightmap_list.append(entropy_heightmap)
            mask_heightmap_list.append(mask_heightmap.reshape([1, mask_heightmap.shape[0], -1]))

        entropy_heightmap = get_entropy_mask(entropy_heightmap_list, mask_heightmap_list)
        entropy_heightmap = torch.from_numpy(entropy_heightmap).float().to(device)
        point_list = creat_patch_conv(heat_map=entropy_heightmap, batch_size=1, device=device, num_patch=1,
            test_time=test_time)

        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        entropy_heightmap = entropy_heightmap.cpu()
        pushes, target_center, push_score = push_generator(point_list, valid_depth_heightmap, entropy_heightmap)

        obj_pts_list_all, obj_class_all = get_instance(mask_list, position_list, class_list, chamLoss)
        if push_score <= 2.17:
            break

        # 3D-->pixel
        best_pix_x = pushes[0]
        best_pix_y = pushes[1]

        primitive_position = [best_pix_x * heightmap_resolution + robot_workspace_limits[0][0],
                              best_pix_y * heightmap_resolution + robot_workspace_limits[1][0],
                              valid_depth_heightmap[best_pix_y][best_pix_x] + robot_workspace_limits[2][0]]
        # 尝试采用中心点完成操作
        end_pix_x = target_center[0]
        end_pix_y = target_center[1]

        pos_robot_end = [end_pix_x * heightmap_resolution + robot_workspace_limits[0][0],
                         end_pix_y * heightmap_resolution + robot_workspace_limits[1][0],
                         valid_depth_heightmap[end_pix_y][end_pix_x] + robot_workspace_limits[2][0]]
        primitive_position_end = get_push_end(primitive_position, pos_robot_end, shift=shift)
        primitive_position = np.array(primitive_position).reshape([1, -1])
        start_list.append(primitive_position)

        primitive_position[:, 2] = primitive_position[:, 2] + 0.019  # 0.018
        primitive_position_end[:, 2] = primitive_position_end[:, 2] + 0.019  # 0.018


        robot.push_exp(target_position=[], start_position=primitive_position[0], end_position=primitive_position_end[0],
            curr_pos=init_pos, mode=1, push_length=shift)

        robot.move_to_v(curr_pos=get_state(ur5_ip), dest_pos=init_pos, angles=0)
        time.sleep(5)
    tmp_list = []
    for class_one in obj_class_all:
        txt_path = os.path.join("./tmp", class_one + ".txt")
        tmp_list.append(np.loadtxt((txt_path)))
    return obj_pts_list_all, obj_class_all, tmp_list, position_list[-1]


def get_pts_from_mask(position, mask):
    data = np.concatenate((position[:, :3], mask.reshape([-1, 1])), axis=1)
    data = data[data[:, 3] == 1]
    return data

def get_cd_distance(source, target, chamLoss, device):
    source_value = source[:, :3].copy()
    target_value = target[:, :3].copy()
    source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
    target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
    source_value = source_value.unsqueeze(0)
    target_value = target_value.unsqueeze(0)
    with torch.no_grad():
        # chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss.cpu().numpy()

def get_view_list(mask_list, mask_list_other, pts, class_list):
    mask_2_4 = []
    obj_pts_2 = []
    pts_2_list = []
    obj_class = []
    for i in mask_list_other:
        mask_2_single = mask_list[i, :, :]
        mask_2_4.append(mask_2_single)
        #pts_one_2 = get_pts_from_mask_filter(pts, i+1)
        pts_one_2 = get_pts_from_mask(pts, mask_2_single)
        if pts_one_2.shape[0] <= 1:
            continue
        obj_pts_2.append(pts_one_2)
        pts_2_list.append(pts_one_2)
        obj_class.append(class_list[i])
    return obj_pts_2, mask_2_4, pts_2_list, obj_class

def get_instance(mask_list_old, pts_list, class_list, chamLoss, device=None):

    # mask_list_old: list mask[0 or 1]
    # class_list: list
    # pts_list: list
    mask_list = copy.deepcopy(mask_list_old)
    num_of_view = len(mask_list)

    mask_top = mask_list[0]
    pts_top = pts_list[0]  # [n, 3]-->[n, x, y, z]
    class_top = class_list[0]
    # 编辑存储的索引
    obj_pts = []
    obj_pts_final = []
    obj_class_final = []

    other_mask_list = []
    pts_top_list = []
    obj_class = []
    other_class_list = []

    for j in range(mask_top.shape[0]):


        pts_one_top = get_pts_from_mask(pts_top, mask_top[j, :, :])
        pts_top_list.append(pts_one_top)

        obj_pts.append(pts_one_top)

        obj_class.append(class_top[j])


    for view_id in range(1, num_of_view):
        mask_single = mask_list[view_id]
        pts_single = pts_list[view_id]
        class_single = class_list[view_id]
        other_mask_id = []
        other_class_id = []
        for i in range(mask_single.shape[0]):
            pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])
            if pts_one.shape[0] <= 10:
                continue
            score = []
            for j in range(mask_top.shape[0]):
                pts_one_top = pts_top_list[j]
                if pts_one_top.shape[0] <= 10:
                    score.append(np.ones(1)[0])
                else:
                    cd_dis = get_cd_distance(pts_one, pts_one_top, chamLoss, device)
                    score.append(cd_dis)

            score = np.asarray(score)
            inds = np.argmin(score)
            score_min = score[inds]
            if score_min > 0.0005:
                other_mask_id.append(i)
                other_class_id.append(class_single[i])
            else:
                obj_pts[inds] = np.concatenate((obj_pts[inds], pts_one), axis=0)

        other_mask_list.append(other_mask_id)
        other_class_list.append(other_class_id)

    obj_pts_2, mask_2, pts_2_list, obj_2_class = get_view_list(mask_list[2], other_mask_list[1], pts_list[2], class_list[2])
    mask_other = mask_2
    obj_pts_other = obj_pts_2
    pts_list_other = pts_2_list
    obj_class_other = obj_2_class

    if len(pts_list_other) > 0:
        other_mask_list_2 = []
        for view_id in [1, 3]:
            if view_id >= num_of_view:
                break
            masks_inds = other_mask_list[view_id - 1]
            mask_single = mask_list[view_id]
            pts_single = pts_list[view_id]
            other_mask_id = []
            for i in masks_inds:
                pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])
                if pts_one.shape[0] <= 10:
                    continue
                score = []
                for j in range(len(mask_other)):
                    pts_one_other = pts_list_other[j]
                    if pts_one_other.shape[0] <= 0:
                        score.append(np.ones(1)[0])
                    else:
                        cd_dis = get_cd_distance(pts_one, pts_one_other, chamLoss, device)
                        score.append(cd_dis)
                score = np.asarray(score)
                inds = np.argmin(score)
                score_min = score[inds]
                if score_min > 0.0005:
                    other_mask_id.append(i)
                else:
                    obj_pts_other[inds] = np.concatenate((obj_pts_other[inds], pts_one), axis=0)
            other_mask_list_2.append(other_mask_id)

        obj_pts_final = []
        obj_class_final = []
        for i in other_mask_list_2[0]:
            obj_pts_final.append(get_pts_from_mask(pts_list[1], mask_list[1][i, :, :]))
            obj_class_final.append(class_list[1][i])
    obj_pts_list_all = obj_pts + obj_pts_other + obj_pts_final
    obj_class_all = obj_class + obj_class_other + obj_class_final

    return obj_pts_list_all, obj_class_all



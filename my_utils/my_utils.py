import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch
import copy
#import chamfer3D.dist_chamfer_3D, fscore
import os
from patch_conv import patch_conv



def init_one_cam(ds5_serial_id):
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 1280×720
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline_one = rs.pipeline()

    config.enable_device(ds5_serial_id)
    pipeline_one.start(config)

    return pipeline_one

def init_mulity_cam(devices_num, ds5_serial_list):


    serial_number_list = ["", "", ""]
    config_list = []
    for i in range(devices_num):
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 1280×720
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_list.append(config)


    connect_device = []
    for d in rs.context().devices:
        print('Found device: ',
            d.get_info(rs.camera_info.name), ' ',
            d.get_info(rs.camera_info.serial_number))
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))

    print(len(connect_device))
    assert (len(connect_device) == devices_num), "wrong"


    pipeline_list = []
    for cam_id in range(devices_num):
        # config_list[cam_id].enable_stream(rs.stream.infrared, cam_id, 640, 480, rs.format.y8, 30) bug?
        pipeline_one = rs.pipeline()
        config_list[cam_id].enable_device(ds5_serial_list[cam_id])
        pipeline_one.start(config_list[cam_id])
        pipeline_list.append(pipeline_one)

    return pipeline_list

def get_current_images(pipeline_list, align):
    color_img_list = []
    depth_img_list = []
    align_depth_frame_list = []

    for i in range(len(pipeline_list)):
        for _ in range(200):
            frames = pipeline_list[i].wait_for_frames()
        aligned_frames = align.process(frames)  # 对齐操作
        aligned_depth_frame = aligned_frames.get_depth_frame()
        align_depth_frame_list.append(aligned_depth_frame)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_img_list.append(color_image)
        depth_img_list.append(depth_image)

    return color_img_list, depth_img_list, align_depth_frame_list

def get_one_image(pipeline, align):
    for _ in range(200):
        frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # 对齐操作
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    return color_image, depth_image, aligned_depth_frame


def entropy_map(pred):
    #entropy_map-->[h, w, cls]-->[640, 1024, 17]-->[640, 1024]
    cls = pred.shape[2]
    #pred = torch.sigmoid(pred)
    nc_log = torch.log(pred)
    nc_entropy = torch.mul(pred, nc_log)
    nc_entropy = torch.where(torch.isnan(nc_entropy), torch.full_like(nc_entropy, 0), nc_entropy)
    #entropy_heat_map = torch.add(torch.sum(nc_map, dim=1), torch.sum(conf_entropy, dim=1))
    entropy_heat_map = torch.sum(nc_entropy, dim=2) / cls
    #entropy_heat_map = torch.sum(conf_entropy, dim=1)
    return -entropy_heat_map

def get_points(depth_img, aligned_depth_frame, mtx, R_cam2base, t_cam2base):

    h, w = depth_img.shape
    idx_proj_mat = np.zeros((h * w, 2), dtype=np.int32)
    depth_heightmap = np.zeros_like(depth_img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            idx_proj_mat[i * w + j] = [j, i]
    idx_proj_mat = idx_proj_mat.reshape((h, w, -1))
    # 配置相机内参
    color_intrin_part = [mtx[0, 2], mtx[1, 2], mtx[0, 0], mtx[1, 1]]  # intrinsics from calibration
    pointcloud = np.zeros((h * w, 3))
    for i in range(h):
        for j in range(w):
            wld_coor = pix2wld(color_intrin_part, aligned_depth_frame,
                idx_proj_mat[i, j], R_cam2base, t_cam2base)
            pointcloud[i * w + j] = wld_coor
            depth_heightmap[i, j] = wld_coor[-1]

    return pointcloud, depth_heightmap

def get_point_e1h(depth_img, aligned_depth_frame, mtx, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base):
    h, w = depth_img.shape
    idx_proj_mat = np.zeros((h * w, 2), dtype=np.int32)
    depth_heightmap = np.zeros_like(depth_img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            idx_proj_mat[i * w + j] = [j, i]
    idx_proj_mat = idx_proj_mat.reshape((h, w, -1))
    # 配置相机内参
    color_intrin_part = [mtx[0, 2], mtx[1, 2], mtx[0, 0], mtx[1, 1]]  # intrinsics from calibration
    pointcloud = np.zeros((h * w, 3))
    for i in range(h):
        for j in range(w):
            wld_coor = pix2wld_e1h(color_intrin_part, aligned_depth_frame,
                idx_proj_mat[i, j], R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
            pointcloud[i * w + j] = wld_coor
            depth_heightmap[i, j] = wld_coor[-1]

    return depth_heightmap, pointcloud

def pix2wld(color_intrin_part, aligned_depth_frame, target_pix, R_cam2base, t_cam2base):
    coor_cam = pix2cam(color_intrin_part, aligned_depth_frame, target_pix)
    coor_base = np.dot(R_cam2base, coor_cam) + t_cam2base

    return coor_base


def pix2wld_e1h(color_intrin_part, aligned_depth_frame, target_pix, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base):
    coor_cam = pix2cam(color_intrin_part, aligned_depth_frame, target_pix)
    coor_gripper = np.dot(R_cam2gripper, coor_cam) + t_cam2gripper
    coor_base = np.dot(R_gripper2base, coor_gripper) + t_gripper2base

    return coor_base

def pix2cam(color_intrin_part, aligned_depth_frame, target_pix):

    ppx = color_intrin_part[0]
    ppy = color_intrin_part[1]
    fx = color_intrin_part[2]
    fy = color_intrin_part[3]

    target_depth = aligned_depth_frame.get_distance(target_pix[0], target_pix[1])

    target_xy_true = [(target_pix[0] - ppx) * target_depth / fx,
                      (target_pix[1] - ppy) * target_depth / fy]

    target_cam = np.array([target_xy_true[0], target_xy_true[1], target_depth])

    return target_cam


def get_pts_from_mask(position, mask):
    data = np.concatenate((position[:, :3], mask.reshape([-1, 1])), axis=1)
    data = data[data[:, 3] == 1]
    return data

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

def get_cd_distance(source, target, device):
    source_value = source[:, :3].copy()
    target_value = target[:, :3].copy()
    source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
    target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
    source_value = source_value.unsqueeze(0)
    target_value = target_value.unsqueeze(0)
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss.cpu().numpy()

def get_instance(mask_list_old, pts_list, class_list, device):
    mask_list = copy.deepcopy(mask_list_old)
    num_of_view = len(mask_list)
    for i in range(len(mask_list)):
        mask = mask_list[i]
        for j in range(mask.shape[0]):
            kernel = np.ones((3, 3), np.uint8) # 投稿(3, 3)
            mask_one = mask[j, :, :]
            erosion_1 = cv2.erode(mask_one, kernel, iterations=1)
            mask[j, :, :] = erosion_1
    mask_top = mask_list[0]  # [n, h, w]
    pts_top = pts_list[0]  # [n, 4]-->[n, x, y, z, 1]
    class_top = class_list[0]
    # 编辑存储的索引
    obj_pts = []
    masks_top_inds = []
    obj_pts_final = []
    obj_class_final = []

    other_mask_list = []
    pts_top_list = []
    obj_class = []
    other_class_list = []

    for j in range(mask_top.shape[0]):
        #pts_one_top = get_pts_from_mask_filter(pts_top, mask_top[j, :, :])
        pts_one_top = get_pts_from_mask(pts_top, mask_top[j, :, :])
        pts_top_list.append(pts_one_top)

        obj_pts.append(pts_one_top)

        obj_class.append(class_top[j])
    for view_id in range(1, num_of_view): # 5
        mask_single = mask_list[view_id]
        pts_single = pts_list[view_id]
        class_single = class_list[view_id]
        other_mask_id = []
        other_class_id = []
        for i in range(mask_single.shape[0]):
            #pts_one = get_pts_from_mask_filter(pts_single, mask_single[i, :, :])
            pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])

            if pts_one.shape[0] <= 0:
                continue
            score = []
            for j in range(mask_top.shape[0]):
                pts_one_top = pts_top_list[j]
                if pts_one_top.shape[0] <= 0:
                    score.append(np.ones(1)[0])
                else:
                    cd_dis = get_cd_distance(pts_one, pts_one_top, device)
                    score.append(cd_dis)

            score = np.asarray(score)
            inds = np.argmin(score)
            score_min = score[inds]
            #print(score_min)
            if score_min > 0.005:
                other_mask_id.append(i)
                other_class_id.append(class_single[i])
            else:
                obj_pts[inds] = np.concatenate((obj_pts[inds], pts_one), axis=0)

        other_mask_list.append(other_mask_id)
        other_class_list.append(other_class_id)

    if num_of_view <= 1:
        pts_list_other = []
        mask_other = []
        obj_pts_other = []
        obj_class_other = []
    elif num_of_view <= 2:
        pts_list_other = []
        obj_pts_other = other_mask_list[0]
        obj_class_other = other_class_list[0]
    elif num_of_view <= 4:
        obj_pts_2, mask_2, pts_2_list, obj_2_class = get_view_list(mask_list[2], other_mask_list[1], pts_list[2], class_list[2])
        mask_other = mask_2
        obj_pts_other = obj_pts_2
        pts_list_other = pts_2_list
        obj_class_other = obj_2_class
    else:
        obj_pts_2, mask_2, pts_2_list, obj_2_class = get_view_list(
            mask_list[2], other_mask_list[1], pts_list[2], class_list[2])
        obj_pts_4, mask_4, pts_4_list, obj_4_class = get_view_list(
            mask_list[4], other_mask_list[3], pts_list[4], class_list[4])
        # obj_pts_1, mask_1, pts_1_list = get_view_list(mask_list[1], other_mask_list[0], pts_list[1])
        # obj_pts_3, mask_3, pts_3_list = get_view_list(mask_list[3], other_mask_list[2], pts_list[3])
        mask_other = mask_2 + mask_4
        obj_pts_other = obj_pts_2 + obj_pts_4
        pts_list_other = pts_2_list + pts_4_list
        obj_class_other = obj_2_class + obj_4_class

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
                #pts_one = get_pts_from_mask_filter(pts_single, mask_single[i, :, :])
                pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])
                # score = np.ones(len(mask_other))
                if pts_one.shape[0] <= 0:
                    continue
                score = []
                for j in range(len(mask_other)):
                    pts_one_other = pts_list_other[j]
                    if pts_one_other.shape[0] <= 0:
                        score.append(np.ones(1)[0])
                    else:
                        cd_dis = get_cd_distance(pts_one, pts_one_other, device)
                        score.append(cd_dis)
                score = np.asarray(score)
                inds = np.argmin(score)
                score_min = score[inds]
                if score_min > 0.005:
                    other_mask_id.append(i)
                else:
                    obj_pts_other[inds] = np.concatenate((obj_pts_other[inds], pts_one), axis=0)

            other_mask_list_2.append(other_mask_id)
        obj_pts_final = []
        obj_class_final = []
        for i in other_mask_list_2[0]:
            #obj_pts_final.append(get_pts_from_mask_filter(pts_list[1], mask_list[1][i, :, :]))
            obj_pts_final.append(get_pts_from_mask(pts_list[1], mask_list[1][i, :, :]))
            obj_class_final.append(class_list[1][i])
        if num_of_view > 3:
            for i in other_mask_list_2[1]:
                #obj_pts_final.append(get_pts_from_mask_filter(pts_list[3], mask_list[3][i, :, :]))
                obj_pts_final.append(get_pts_from_mask(pts_list[3], mask_list[3][i, :, :]))
                obj_class_final.append(class_list[3][i])
    obj_pts_list_all = obj_pts + obj_pts_other + obj_pts_final
    obj_class_all = obj_class + obj_class_other + obj_class_final
    #print(len(obj_pts_list_all))
    #print(len(obj_class_all))
    return obj_pts_list_all, obj_class_all

def get_patch(img, img_xy, patch_size=200):
    x1 = img_xy[0]
    y1 = img_xy[1]
    patch = img[y1:y1+patch_size, x1:x1+patch_size]
    return patch

def get_shift(start_list, shift):
    start_point_1 = start_list[-2]
    start_point_2 = start_list[-1]
    x1 = start_point_1[0, 0]
    y1 = start_point_1[0, 1]
    x2 = start_point_2[0, 0]
    y2 = start_point_2[0, 1]
    distance = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    if distance < 0.05:
        shift = shift + 0.05
    return shift

def data_normal_2d(orign_data, dim="col"):

    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    return norm_data

def get_heightmap(surface_pts, workspace_limits, segment_heat_map, segment_results, heightmap_resolution=0.002):

    entropy_pts = segment_heat_map.reshape([-1, 1])
    mask_pts = segment_results.reshape([-1, 1])
    # Compute heightmap size

    heightmap_size = np.ceil(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    # Sort surface points by z value

    sort_z_ind = np.argsort(surface_pts[:, 2])
    entropy_pts = entropy_pts[sort_z_ind]
    mask_pts = mask_pts[sort_z_ind]
    surface_pts = surface_pts[sort_z_ind]
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] > workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] > workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
                                         surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    entropy_pts = entropy_pts[heightmap_valid_ind]
    mask_pts = mask_pts[heightmap_valid_ind]
    entropy_heightmap = np.zeros((heightmap_size[0], heightmap_size[1], 1))
    mask_heightmap = np.zeros((heightmap_size[0], heightmap_size[1], 1))
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    entropy_heightmap[heightmap_pix_y, heightmap_pix_x] = entropy_pts[:, [0]]
    mask_heightmap[heightmap_pix_y, heightmap_pix_x] = mask_pts[:, [0]]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return depth_heightmap, entropy_heightmap.reshape([entropy_heightmap.shape[0], -1]), mask_heightmap.reshape([mask_heightmap.shape[0], -1])

def get_heightmap_only(surface_pts, workspace_limits, segment_heat_map, segment_results, heightmap_resolution=0.002):

    # Compute heightmap size
    heightmap_size = np.ceil(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] > workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] > workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
                                         surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]

    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]

    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return depth_heightmap

def get_bullet_heightmap_only(surface_pts, workspace_limits, heightmap_resolution):
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    # Get 3D point cloud from RGB-D images
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    return depth_heightmap

def get_bullet_heightmap(surface_pts, masks_imgs, workspace_limits, heightmap_resolution):
    num_masks = masks_imgs.shape[2]
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    # Get 3D point cloud from RGB-D images
    masks_pts = masks_imgs.copy()
    masks_pts = masks_pts.transpose(2, 0, 1).reshape(num_masks, -1)
    print(masks_pts.shape)
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    masks_pts = masks_pts[:, sort_z_ind]
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    masks_pts = masks_pts[:, heightmap_valid_ind]
    masks_heightmaps = np.zeros((heightmap_size[0], heightmap_size[1], num_masks), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    for c in range(num_masks):
        masks_heightmaps[heightmap_pix_y, heightmap_pix_x, c] = masks_pts[c, :]
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return depth_heightmap, masks_heightmaps



def get_entropy_mask(entropy_heightmap_list, mask_heightmap_list, time = 0):

    num_of_mask = len(mask_heightmap_list)
    # entropy_heightmap_list 相加
    entropy_heightmap = np.zeros_like(entropy_heightmap_list[0])
    for i in range(len(entropy_heightmap_list)):
        #path = "/home/wzy/Desktop/pybullet-object-models-master/ec_map/"
        entropy_heightmap = entropy_heightmap + entropy_heightmap_list[i]
    # base_path = "/home/wzy/Desktop/pybullet-object-models-master/ec_map/"
    # entropy_path = os.path.join(base_path, "entropy" + "_" + str(time) + '.png')
    # show_ec_map_numpy(entropy_heightmap, entropy_path)
    """
    # entropy_heightmap_list 相乘
    entropy_heightmap = np.ones_like(entropy_heightmap_list[0])
    for i in range(len(entropy_heightmap_list)):
        entropy_heightmap = np.multiply(entropy_heightmap, entropy_heightmap_list[i])
    """
    # 处理mask, 按照列来获取
    mask_heightmap_all = np.concatenate(mask_heightmap_list, axis=0)
    mask_heightmap_all = mask_heightmap_all.reshape([num_of_mask, -1]) # 5
    mask_heightmap = np.zeros([mask_heightmap_all.shape[1]])
    for i in range(mask_heightmap_all.shape[1]):
        mask_heightmap[i] = len(np.unique( mask_heightmap_all[:, i]))
    mask_heightmap = mask_heightmap + 0.0001
    mask_heightmap = mask_heightmap.reshape([-1, entropy_heightmap.shape[1]])
    # mask_path = os.path.join(base_path, "mask" + "_" + str(time) + '.png')
    # show_numpy_map(mask_heightmap, "object")
    np.save("object.npy", mask_heightmap)
    entropy_heightmap = np.multiply(entropy_heightmap, mask_heightmap)
    return entropy_heightmap

def creat_patch_conv(heat_map, batch_size, device, num_patch=1, test_time=1, kernal_size=200): # 75
    patch_list = []
    patch_select = []
    heat_map = heat_map.view(1, -1, heat_map.shape[0], heat_map.shape[1])
    patch_sum = patch_conv(kernal_size=kernal_size).to(device)
    # patch_sum.half()
    out = patch_sum(heat_map)
    # show_tensor_map(out, "patch_entropy")
    np.save("entropy.npy", out.view(-1, out.size(-1)).detach().cpu().numpy().astype(np.int16))
    print(out.shape)
    if kernal_size < 100:
        base_path_3 = "/home/wzy/Desktop/pybullet-object-models-master/ec_map/"
        #img_path_3 = os.path.join(base_path_3, "entropy_all" + "_"+ str(test_time) + '.png')
        #show_ec_map(out, img_path_3)
    #path = "/home/wzy/Desktop/pybullet-object-models-master/ec_map/entropy.png"
    #show_ec_map(out, path)
    out = out.float()
    out_t = out.flatten()
    sorted, _ = torch.sort(out_t)
    out = out.cpu().detach()
    sorted = sorted.cpu().detach().numpy()
    sorted = sorted[::-1]
    patch_xy = np.zeros([1, 2])
    ind = np.argwhere(out == sorted[0])
    print("max_entropy", sorted[0])

    x1 = int(ind[3, 0])
    y1 = int(ind[2, 0])
    patch_list.append([x1, y1])
    #ind = np.argwhere(out == sorted[-1])
    #print("min_entropy", sorted[-1])
    #x2 = int(ind[3, 0])
    #y2 = int(ind[2, 0])
    #patch_list.append([int(x1+kernal_size/2), int(y1+kernal_size/2)])
    return patch_list

def push_generator(point_list, valid_depth_heightmap, entropy_heightmap, task=0):

    area_shape_default = [200, 200]
    H, W = valid_depth_heightmap.shape
    target_center = (int(point_list[0][0]+area_shape_default[0]/2), int(point_list[0][1]+area_shape_default[0]/2))
    #print((H, W))
    #print(valid_depth_heightmap.shape)

    if task == 0:
        #height_target = valid_depth_heightmap[target_center[1], target_center[0]]
        height_target = get_average_height(valid_depth_heightmap, target_center, area_shape_default)
        print(height_target)
        #area_shape_default = [100, 100]

    else:
        #height_target = get_average_height(valid_depth_heightmap, target_center, area_shape_default)
        #height_target = get_max_height(valid_depth_heightmap, target_center, area_shape_default)
        height_target = valid_depth_heightmap[target_center[1], target_center[0]]
        area_shape_default = [100, 100]
    target_center_x = target_center[0]
    target_center_y = target_center[1]
    # ROI具体范围
    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0] / 2),
                                min(target_center_x + area_shape_default[0] / 2, W-1))
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1] / 2),
                                min(target_center_y + area_shape_default[1] / 2, H-1))
    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int)
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2))
    # Searching for suitable starting points as candidates
    # 注意: det_inds_mat为[x, y]
    det_inds_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y), 2))
    det_height_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y)))
    push_ind_mat = area_indices.copy()
    vshift = 6
    hshift = 6
    push_score = 0
    translations = np.zeros((2, 2, 2))
    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(2 * j - 1) * hshift, (2 * i - 1) * vshift]
    for i in range(2):
        for j in range(2):
            det_inds_mat[2 * i + j, :] = np.floor((push_ind_mat + translations[i, j]))
    det_inds_mat = det_inds_mat.astype(np.int)
    for i in range(2):
        for j in range(2):
            idx = tuple([det_inds_mat[2 * i + j, :, 1], det_inds_mat[2 * i + j, :, 0]])
            idx[0][idx[0] >= H] = H-1
            idx[0][idx[0] < 0] = 0
            idx[1][idx[1] >= W] = W-1
            idx[1][idx[1] < 0] = 0
            det_height_mat[2 * i + j, :] = valid_depth_heightmap[idx]
    height_det = np.max(det_height_mat, axis=0)
    valid = (height_target - height_det >= 0.015)
    # height_score
    height_score_map= height_target - valid_depth_heightmap
    #height_score = height_score.reshape([area_shape_default[0], -1])

    valid_push_inds = np.where(valid == True)[0].flatten()
    num_valid_push = valid_push_inds.shape[0]
    if num_valid_push >= 1:
        quadrants = [[], [], [], []]
        valid_push_ids = area_indices[valid_push_inds]
        for idx in valid_push_ids:
            if idx[0] <= target_center[0] and idx[1] <= target_center[1]:
                quadrants[0].append(idx)
            elif idx[0] > target_center[0] and idx[1] <= target_center[1]:
                quadrants[1].append(idx)
            elif idx[0] <= target_center[0] and idx[1] > target_center[1]:
                quadrants[2].append(idx)
            elif idx[0] > target_center[0] and idx[1] > target_center[1]:
                quadrants[3].append(idx)

        candidates = []
        for sector in quadrants:
            #if len(sector) > 10:
            #    sampled_push = random.sample(sector, 10)
            #    candidates.append(sampled_push)
            #else:
            candidates.append(sector)
        all_candidates = []
        all_candidates_entropy = []
        all_candidates_height = []
        for sector in candidates:
            for start_point in sector:
                all_candidates.append(start_point)
                x0 = start_point[0]
                y0 = start_point[1]
                all_candidates_entropy.append(entropy_heightmap[y0][x0])
                # [9, 2]
                #rowIndex = np.where((area_indices == [y0, x0]).all(axis=1))
                all_candidates_height.append(height_score_map[y0][x0])

        all_candidates_entropy = np.asarray(all_candidates_entropy)
        all_candidates_height = np.asarray(all_candidates_height)
        all_score = np.multiply(all_candidates_height, all_candidates_entropy)
        id_max = np.argsort(all_score)[-1]
        push_score = all_score[id_max]
        print("push得分", all_score[id_max])
        push_start_point = all_candidates[id_max]
    else:
        print("SCT faile")
        if target_center[0] < W/2:
            push_start_point = [np.max((target_center[0] - int(area_shape_default[0]/4), 0)), target_center[1]]
        else:
            push_start_point = [np.min((target_center[0] + int(area_shape_default[0]/4), W-1)), target_center[1]]
        push_start_point = np.array(push_start_point)
    return push_start_point, target_center, push_score

def get_push_end(start_point, end_point, shift=0.15):
    # start_point: [x, y, z]
    # end_point: [[x, y, z]]
    x1 = start_point[0]
    y1 = start_point[1]
    z1 = start_point[2]
    x2 = end_point[0]
    y2 = end_point[1]
    z2 = end_point[2]
    r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) + 0.0001
    end_y = (shift*(y2-y1))/r + y1
    end_x = (shift*(x2-x1))/r + x1
    end_point = np.array([[end_x, end_y, z1]])
    return end_point

def get_average_height(valid_depth_heightmap, target_center, area_shape_default):
    H, W = valid_depth_heightmap.shape
    target_center_x = target_center[0]
    target_center_y = target_center[1]
    x1 = max(0, target_center_x - area_shape_default[0] / 2)
    y1 = max(0, target_center_y - area_shape_default[1] / 2)
    x2 = min(target_center_x + area_shape_default[0] / 2, W - 1)
    y2 = min(target_center_y + area_shape_default[1] / 2, H - 1)
    height = np.mean(valid_depth_heightmap[int(y1):int(y2), int(x1):int(x2)])
    return height

def get_max_height(valid_depth_heightmap, target_center, area_shape_default):
    H, W = valid_depth_heightmap.shape
    target_center_x = target_center[0]
    target_center_y = target_center[1]
    x1 = max(0, target_center_x - area_shape_default[0] / 2)
    y1 = max(0, target_center_y - area_shape_default[1] / 2)
    x2 = min(target_center_x + area_shape_default[0] / 2, W - 1)
    y2 = min(target_center_y + area_shape_default[1] / 2, H - 1)
    height = np.max(valid_depth_heightmap[int(y1):int(y2), int(x1):int(x2)])
    return height

def show_tensor_map(entropy_map, name):

    plt.matshow(entropy_map.view(-1, entropy_map.size(-1)).detach().cpu().numpy().astype(np.int16), interpolation='nearest')
    # plt.matshow(entropy_map.astype(np.int16), interpolation='nearest')
    # plt.show()
    base_path = os.getcwd()
    path = os.path.join(base_path, name + ".png")
    plt.savefig(path, transparent=True, dpi=800)
    plt.close()
def show_numpy_map(map, name):
    plt.matshow(map.astype(np.int16), interpolation='nearest')
    base_path = os.getcwd()
    path = os.path.join(base_path, name + ".png")
    plt.savefig(path, transparent=True, dpi=800)
    plt.close()
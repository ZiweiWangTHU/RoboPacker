import numpy as np
import cv2
import time
import random

# Set parameters
finger_width = 0.02
heightmap_resolution = 0.002
gripper_open_width_inner = 0.12
gripper_open_width_outter = 0.14
num_det_samples = int(finger_width / heightmap_resolution)


def get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits):

    # np.save('valid_depth_heightmap.npy', valid_depth_heightmap)

    points = np.zeros((valid_depth_heightmap.shape[0], valid_depth_heightmap.shape[1], 3))
    for x in range(valid_depth_heightmap.shape[1]):
        for y in range(valid_depth_heightmap.shape[0]):
            points[y][x] = [x * heightmap_resolution + workspace_limits[0][0],
                            y * heightmap_resolution + workspace_limits[1][0],
                            valid_depth_heightmap[y][x] + workspace_limits[2][0]]
    pointcloud = points.reshape(-1, 3)
    return pointcloud


def gripper_vis(grasp_points):
    pc_resolution = 0.001

    point = np.zeros(3)
    grippers_pc = np.zeros(3)
    for i, grasp in enumerate(grasp_points):
        rot_ind = grasp[0]
        grasp_point = grasp[1]
        center = grasp_point.copy()

        center[2] += 0.05
        theta = np.pi / 16 * rot_ind
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        part_1 = np.zeros((10*100*10, 3))
        idx = 0
        for i in range(6):
            for j in range(100):
                for k in range(6):
                    point[0] = center[0] - 0.003 + i*pc_resolution
                    point[1] = center[1] - 0.05 + j*pc_resolution
                    point[2] = center[2] - 0.003 + k*pc_resolution
                    part_1[idx] = point
                    idx += 1

        part_2 = np.zeros((10 * 10 * 40, 3))
        idx = 0
        for i in range(6):
            for j in range(6):
                for k in range(40):
                    point[0] = center[0] - 0.003 + i * pc_resolution
                    point[1] = center[1] - 0.05 + j * pc_resolution
                    point[2] = center[2] - 0.04 + k * pc_resolution
                    part_2[idx] = point
                    idx += 1

        part_3 = np.zeros((10 * 10 * 40, 3))
        idx = 0
        for i in range(6):
            for j in range(6):
                for k in range(40):
                    point[0] = center[0] - 0.003 + i * pc_resolution
                    point[1] = center[1] + 0.044 + j * pc_resolution
                    point[2] = center[2] - 0.04 + k * pc_resolution
                    part_3[idx] = point
                    idx += 1

        gripper_pc = np.vstack((part_1, part_2, part_3))
        gripper_xy = gripper_pc[:, 0:2]
        trasnslations_xy = gripper_xy - center[0:2]
        trasnslations_xy_rot = np.matmul(trasnslations_xy, R)
        gripper_xy_rot = trasnslations_xy_rot + center[0:2]
        gripper_pc[:, 0: 2] = gripper_xy_rot
        # print(gripper.shape)
        # grippers_pc.append(gripper)
        grippers_pc = np.vstack((grippers_pc, gripper_pc))

        with open('sample_grippers_points.txt', 'w') as f:
            for point in grippers_pc:
                f.write(
                    str(np.float(point[0])) + ';' +
                    str(np.float(point[1])) + ';' +
                    str(np.float(point[2])) + ';' +
                    str(np.int(255)) + ';' +
                    str(np.int(0)) + ';' +
                    str(np.int(0)) + '\n'
                )

    return grippers_pc


def area_mask(pointcloud):
    pointcloud = pointcloud.reshape((224, 224, 3))
    peak_ind = np.where(pointcloud[:, :, 2] == np.max(pointcloud[:, :, 2]))

    target_center_x = peak_ind[0][0]
    target_center_y = peak_ind[1][0]

    proposal_area_x = np.arange(max(0, target_center_x - 56), min(target_center_x + 56, 223))
    proposal_area_y = np.arange(max(0, target_center_y - 56), min(target_center_y + 56, 223))

    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int)
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]

    area_indices = area_indices.reshape((-1, 2))

    mask_cloud = np.zeros((len(area_indices), 3))
    pointcloud = pointcloud.reshape((-1, 3))

    for i, idx in enumerate(area_indices):
        mask_cloud[i] = pointcloud[idx[0] * 224 + idx[1]]

    with open('area_mask.txt', 'w') as f:
        for i, point in enumerate(mask_cloud):
            f.write(
                str(np.float(point[0])) + ';' +
                str(np.float(point[1])) + ';' +
                str(np.float(point[2])) + ';' +
                str(np.int(255)) + ';' +
                str(np.int(0)) + ';' +
                str(np.int(0)) + '\n'
            )


def grasps_generator(target_center, pointcloud_reshaped, valid_depth_heightmap):

    # pointcloud_reshaped = pointcloud.reshape((224, 224, 3)) # correct

    # area_shape_default = [100, 100]

    H = pointcloud_reshaped.shape[0]
    W = pointcloud_reshaped.shape[1]
    # area_shape_default = [112, 112] # [H/2, W/2]
    area_shape_default = [H/2, W/2]
    target_center_x = target_center[0] # correct
    target_center_y = target_center[1]

    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0]/2), min(target_center_x + area_shape_default[0]/2, W-1)) # correct
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1]/2), min(target_center_y + area_shape_default[1]/2, H-1))

    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int) # correct
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2)) # (12544, 2)

    # Get translation matrices of height detection points
    grasp_ind_mat = area_indices.copy()
    hshift = 6 # 6
    # vshift = 24
    vshift = 40
    finger_thickness = 10
    det_inds_mat = np.zeros((5*3, len(proposal_area_x)*len(proposal_area_y), 2))
    det_height_mat = np.zeros((5*3, len(proposal_area_x)*len(proposal_area_y)))
    translations = np.zeros((5, 3, 2))

    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(j - 1) * hshift, (i - 2) * vshift]  # x, y
    translations[0, :, 1] = translations[1, :, 1] - finger_thickness
    translations[4, :, 1] = translations[3, :, 1] + finger_thickness
    # Get valid grasp point indices with different rotations
    grasps = []
    for k in range(16):
        theta = np.pi / 16 * k
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        translations_rot = np.matmul(translations, R).astype(np.int)

        # Get detection point indices with different rotations
        for i in range(5):
            for j in range(3):
                det_inds_mat[3*i+j, :] = np.floor((grasp_ind_mat + translations_rot[i, j]))
        det_inds_mat = det_inds_mat.astype(np.int)
        # print(det_inds_mat.shape)
        # Show height detect points in heightmap
        # if k == 4:
        #     heightmap = valid_depth_heightmap.copy()
        #     heightmap[tuple([det_inds_mat[:, 50: 51, 1], det_inds_mat[:, 50: 51, 0]])] = 1
        #     heightmap[tuple([det_inds_mat[0:9, 50: 51, 1], det_inds_mat[0:9, 50: 51, 0]])] = 1
        #     cv2.imshow('detect points', np.flip(heightmap, 0))
        #     cv2.waitKey(0)

        # Get height data of detection points
        for i in range(5):
            for j in range(3):
                idx = tuple([det_inds_mat[3 * i + j, :, 1], det_inds_mat[3 * i + j, :, 0]])
                idx[0][idx[0] > H - 1] = H - 1
                idx[0][idx[0] < 0] = 0
                idx[1][idx[1] > W - 1] = W - 1
                idx[1][idx[1] < 0] = 0
                det_height_mat[3 * i + j, :] = valid_depth_heightmap[idx]

        # print(det_height_mat.shape)
        # heightmap = valid_depth_heightmap.copy()
        # heightmap[tuple([det_inds_mat[6:16, 50: 51, 1], det_inds_mat[6:16, 50: 51, 0]])] = 1
        # cv2.imshow('detect points', heightmap)
        # cv2.waitKey(0)

        height_det_grasp = np.vstack((det_height_mat[6], det_height_mat[7], det_height_mat[8])) # height of the potential grasp point area
        height_det1 = np.vstack((det_height_mat[0], det_height_mat[1], det_height_mat[2], det_height_mat[3], det_height_mat[4], det_height_mat[5])) # height of the left finger area
        height_det2 = np.vstack((det_height_mat[9], det_height_mat[10], det_height_mat[11], det_height_mat[12], det_height_mat[13], det_height_mat[14])) # height of the right finger area

        height_grasp_min = np.min(height_det_grasp, axis=0)
        height_det1_max = np.max(height_det1, axis=0)
        height_det2_max = np.max(height_det2, axis=0)

        # valid = (height_grasp_min - height_det2_max > 0.025) & (height_grasp_min - height_det1_max > 0.025)
        # valid = (height_grasp_min - height_det2_max > 0.015) & (height_grasp_min - height_det1_max > 0.015)
        valid = (height_grasp_min - height_det2_max > 0.03) & (height_grasp_min - height_det1_max > 0.008)
        valid_grasp_ind = np.where(valid==True)[0].flatten()

        valid_grasp_inds = area_indices[valid_grasp_ind] # Get the indices of the valid grasps
        # print('***', det_height_mat[:, valid_grasp_ind[5]].reshape((5, 3)))
        # print(len(valid_grasp_ind))
        # Show the valid grasps area in heightmap
        # valid_depth_heightmap[tuple([valid_grasp_inds[:, 1], valid_grasp_inds[:, 0]])] = 1
        # theta = np.pi / 16 * k
        # R = np.array([[np.cos(theta), -np.sin(theta)],
        #               [np.sin(theta), np.cos(theta)]])
        # indicator = []
        # for j in range(-35, 35):
        #     print(peak_ind)
        #     idx = [peak_ind[0], peak_ind[1] + j]
        #     indicator.append(idx)
        # print(indicator)
        # indicator = np.asarray(indicator)
        # trans = indicator - peak_ind
        # trans_rot = np.matmul(trans, R).astype(np.int)
        # indicator_rot = trans_rot + peak_ind
        # valid_depth_heightmap[tuple([indicator_rot[:, 1], indicator_rot[:, 0]])] = 0.6
        #
        # cv2.imshow('', valid_depth_heightmap*5)
        # cv2.waitKey(0)

        # Get 3d coordinates of grasp points
        # grasps.append([k, pointcloud[tuple([valid_grasp_inds[:, 0], valid_grasp_inds[:, 1]])]])
        # print('%%%', valid_grasp_inds.shape)

        grasps.append([k, pointcloud_reshaped[tuple([valid_grasp_inds[:, 0], valid_grasp_inds[:, 1]])], valid_grasp_inds])

        # if k == 15:
        #     with open('grasps.txt', 'w') as f:
        #         for i, point in enumerate(grasps):
        #             f.write(
        #                 str(np.float(point[0])) + ';' +
        #                 str(np.float(point[1])) + ';' +
        #                 str(np.float(point[2])) + ';' +
        #                 str(np.int(255)) + ';' +
        #                 str(np.int(0)) + ';' +
        #                 str(np.int(0)) + '\n'
        #             )

    num_valid_grasps = 0
    for i in range(len(grasps)):
        num_valid_grasps += len(grasps[i][1])
    # print('num_valid_grasps: ', num_valid_grasps)

    # Sample a number of grasps for visualization
    # sample_rotations = random.sample(grasps, 8)
    # sample_grasps = []
    # for i in range(len(sample_rotations)):
    #     if len(sample_rotations[i][1]) != 0:
    #         sample_grasp = random.choice(sample_rotations[i][1])
    #         sample_grasps.append([sample_rotations[i][0], sample_grasp])

    # generate grasp_masks
    grasp_mask_heightmaps = []
    # print('***', np.array(grasps).shape)
    for rot_idx in np.array(grasps, dtype=object)[:, 0]:
        for grasp_ind in np.array(grasps, dtype=object)[rot_idx, 2]:
            heightmap = grasp_mask_generator(rot_idx, grasp_ind, H=H, W=W)
            grasp_mask_heightmaps.append([heightmap, grasp_ind, rot_idx])
    grasp_mask_heightmaps = np.asarray(grasp_mask_heightmaps, dtype=object)
    # print('grasp_mask_heightmaps', grasp_mask_heightmaps.shape)
    return grasps, grasp_mask_heightmaps, num_valid_grasps


def grasp_mask_generator(rot_idx, grasp_ind, H=224, W=224):
    hshift = 8
    vshift = 24
    # vshift = 12

    theta = np.pi / 16 * rot_idx
    # grasp_mask_heightmap = np.zeros((224, 224), dtype='uint8')
    grasp_mask_heightmap = np.zeros((H, W), dtype='uint8')
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    trans_x, trans_y = np.mgrid[-hshift: hshift+1: 1, -vshift: vshift+1: 1]
    tras_indices = np.c_[trans_x.ravel(), trans_y.ravel()]
    trans_rot_indices = np.floor(np.matmul(tras_indices, R)).astype(np.int)
    grasp_indices = trans_rot_indices + grasp_ind
    grasp_indices[grasp_indices > H-1] = H-1
    grasp_indices[grasp_indices < 0] = 0

    grasp_mask_heightmap[tuple([grasp_indices[:, 1], grasp_indices[:, 0]])] = 1

    return grasp_mask_heightmap


def push_mask_generator(push_start_point, target_center, rot_ind, H=224, W=224):

    mask_length = 80
    mask_width = 10
    # push_mask = np.zeros((224, 224))
    push_mask = np.zeros((H, W))
    translations = np.zeros((mask_length, mask_width, 2)).astype(np.int)
    if push_start_point[1] < target_center[1]:
        for i in range(0, mask_width):
            for j in range(0, mask_length):
                translations[j, i, :] = [i - mask_width/2, j]
    else:
        for i in range(0, mask_width):
            for j in range(0, mask_length):
                translations[j, i, :] = [i - mask_width/2, -j]

    push_vec = [target_center[0] - push_start_point[0], target_center[1] - push_start_point[1] + 1e-8]
    push_rot_angle = np.arctan(push_vec[0] / push_vec[1])
    rot_angle = np.pi / 8 * (rot_ind - 1)
    rot_angle_final = push_rot_angle + rot_angle
    rot_mat = np.array([[np.cos(rot_angle_final), -np.sin(rot_angle_final)],
                        [np.sin(rot_angle_final), np.cos(rot_angle_final)]])

    translations_rot = np.dot(translations, rot_mat).astype(np.int)

    push_area_inds = push_start_point + translations_rot
    push_area_inds[np.where(push_area_inds > W-1)] = W-1
    push_area_inds[np.where(push_area_inds < 0)] = 0
    push_mask[tuple([push_area_inds[0: int(mask_length/2), :, 1], push_area_inds[0: int(mask_length/2), :, 0]])] = 0.5
    push_mask[tuple([push_area_inds[int(mask_length/2): mask_length, :, 1], push_area_inds[int(mask_length/2): mask_length, :, 0]])] = 1

    push_end_point = push_area_inds[mask_length-1, np.int(mask_width/2)]
    return push_mask, push_end_point


def push_generator(target_center, valid_depth_heightmap):

    area_shape_default = [100, 100]
    height_target = valid_depth_heightmap[target_center[1], target_center[0]]
    H, W = valid_depth_heightmap.shape
    target_center_x = target_center[0]
    target_center_y = target_center[1]

    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0] / 2),
                                min(target_center_x + area_shape_default[0] / 2, W-1))
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1] / 2),
                                min(target_center_y + area_shape_default[1] / 2, H-1))

    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int)
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2))  # default (2500, 2)

    # Searching for suitable starting points as candidates
    det_inds_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y), 2))
    det_height_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y)))
    push_ind_mat = area_indices.copy()
    vshift = 6
    hshift = 6
    translations = np.zeros((2, 2, 2))
    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(2*j - 1) * hshift, (2*i - 1) * vshift]
    for i in range(2):
        for j in range(2):
            det_inds_mat[2 * i + j, :] = np.floor((push_ind_mat + translations[i, j]))
    det_inds_mat = det_inds_mat.astype(np.int)

    for i in range(2):
        for j in range(2):
            idx = tuple([det_inds_mat[2 * i + j, :, 1], det_inds_mat[2 * i + j, :, 0]])
            idx[0][idx[0] > H - 1] = H - 1
            idx[0][idx[0] < 0] = 0
            idx[1][idx[1] > W - 1] = W - 1
            idx[1][idx[1] < 0] = 0
            det_height_mat[2 * i + j, :] = valid_depth_heightmap[idx]

    height_det = np.max(det_height_mat, axis=0)
    valid = (height_target - height_det >= 0.02)
    valid_push_inds = np.where(valid == True)[0].flatten()
    num_valid_push = valid_push_inds.shape[0]
    # print('num_valid_push: ', num_valid_push)

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
            if len(sector) > 10:
                sampled_push = random.sample(sector, 10)
                candidates.append(sampled_push)
            else:
                candidates.append(sector)

        pushes = []
        for sector in candidates:
            for start_point in sector:
                push = []
                rot_ind = np.random.randint(0, 3)
                push_mask, end_point = push_mask_generator(start_point, target_center, rot_ind, H=H, W=W)
                push.append(start_point)
                push.append(rot_ind)
                push.append(push_mask)
                push.append(end_point)
                pushes.append(push)

    else:
        if target_center[0] < 112:
            start_point = [np.max((target_center[0] - 25, 0)), target_center[1]]
        else:
            start_point = [np.min((target_center[0] + 25, 223)), target_center[1]]
        rot_ind = np.random.randint(0, 3)
        push_mask, end_point = push_mask_generator(start_point, target_center, rot_ind, H=H, W=W)
        pushes = [[start_point, rot_ind, push_mask, end_point]]

    return pushes


# Testing code

# target_center = [150, 150]
# push_start_point = [100, 50]
# valid_depth_heightmap = np.load('valid_depth_heightmap.npy')
# push_generator(target_center, valid_depth_heightmap)
# push_mask , end_point= push_mask_generator(push_start_point, target_center, 1)
# cv2.imshow('', push_mask)
# cv2.waitKey(0)

# valid_depth_heightmap = np.load('valid_depth_heightmap.npy')
# workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
# get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits)
# with open('point_cloud.txt', 'r') as f:
#     lines = f.readlines()
#     pointcloud = np.zeros((len(lines), 3))
#     for i, line in enumerate(lines):
#         line = line.strip('\n')
#         data = np.array(line.split(';'), dtype=np.float)
#         pointcloud[i] = data
# pointcloud_reshaped = np.reshape(pointcloud, (224, 224, 3)) # (224, 224, 3)
# peak_ind = np.argmax(pointcloud[:, 2]) # pointcloud shape (50176, 3)
# peak_ind_reshaped = np.unravel_index(np.argmax(pointcloud_reshaped[:, :, 2]), (224, 224))
# peak = pointcloud[peak_ind]
# target_center = [100, 100]
#
#
# push_generator(peak_ind_reshaped, pointcloud)
# grasps, grasp_mask_heightmaps, num_grasps = grasps_generator(target_center, pointcloud)
# # print(num_grasps)
# # print(np.asarray(grasps).shape)
#
# sample_rotations = random.sample(grasps, 8)
# sample_grasps = []
# for i in range(len(sample_rotations)):
#     if len(sample_rotations[i][1]) != 0:
#         sample_grasp = random.choice(sample_rotations[i][1])
#         sample_grasps.append([sample_rotations[i][0], sample_grasp])
#
# sample_grasps = np.asarray(sample_grasps)
# print(sample_grasps.shape)
# gripper_vis(sample_grasps)
# from trainer2 import Trainer

# trainer = Trainer(0.5, False, False, None, False)
# depth_heightmap = np.load('valid_depth_heightmap.npy')
# target_mask_heightmap = np.load('valid_depth_heightmap.npy')
# target_mask_heightmap = np.flip(target_mask_heightmap)
# time0 = time.time()
# confs = []
# grasp_inds = []
# rot_inds = []
#
# sampled_inds = np.random.choice(np.arange(num_grasps), 100, replace=False)
# for i in sampled_inds:
#     grasp_mask_heightmap = grasp_mask_heightmaps[i][0]
#     print('input heightmaps')
#     print(np.max(depth_heightmap), np.min(depth_heightmap), np.mean(depth_heightmap))
#     print(np.max(target_mask_heightmap), np.min(target_mask_heightmap), np.mean(target_mask_heightmap))
#     print(np.max(grasp_mask_heightmap), np.min(grasp_mask_heightmap), np.mean(grasp_mask_heightmap))
#     confidence, _ = trainer.forward(depth_heightmap, target_mask_heightmap, grasp_mask_heightmap)
#
#     confs.append(confidence)
#     grasp_inds.append(grasp_mask_heightmaps[i][1])
#     rot_inds.append(grasp_mask_heightmaps[i][2])
#
# grasp_inds = np.hstack((np.array(rot_inds).reshape((-1, 1)), np.array(grasp_inds)))
#
# best_conf = np.max(confs)
# best_ind = np.argmax(confs)
# best_grasp_ind = grasp_inds[best_ind]
# print(best_grasp_ind)




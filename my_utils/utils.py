import math

import numpy as np
from skimage.morphology.convex_hull import convex_hull_image
from scipy.ndimage.morphology import binary_dilation


def check_grasp_margin(target_mask_heightmap, depth_heightmap):
    margin_mask = binary_dilation(target_mask_heightmap, iterations=10).astype(np.float32)-target_mask_heightmap
    margin_depth = margin_mask * depth_heightmap
    margin_depth[np.isnan(margin_depth)] = 0
    margin_depth[margin_depth > 0.3] = 0
    margin_depth[margin_depth < 0.02] = 0
    margin_depth[margin_depth > 0] = 1
    margin_value = np.sum(margin_depth)
    return margin_value/np.sum(margin_mask), margin_value/np.sum(target_mask_heightmap)


def check_push_target_oriented(best_pix_ind, push_end_pix_yx, target_mask_heightmap, mask_count_threshold=5):
    mask_hull = convex_hull_image(target_mask_heightmap)
    mask_count = 0
    x1 = best_pix_ind[2]
    y1 = best_pix_ind[1]
    x2 = push_end_pix_yx[1]
    y2 = push_end_pix_yx[0]
    x_range = abs(x2-x1)
    y_range = abs(y2-y1)
    if x_range > y_range:
        k = (y2-y1)/(x2-x1)
        b = y1-k*x1
        for x in range(min(int(x1), int(x2)), max(int(x1), int(x2))+1):
            y = int(k*x+b)
            try:
                mask_count += mask_hull[y, x]
            except IndexError:
                pass
    else:
        k = (x2-x1)/(y2-y1)
        b = x1-k*y1
        for y in range(min(int(y1), int(y2)), max(int(y1), int(y2))+1):
            x = int(k*y+b)
            try:
                mask_count += mask_hull[y, x]
            except IndexError:
                pass
    if mask_count > mask_count_threshold:
        return True
    else:
        return False


def check_grasp_target_oriented(best_pix_ind, target_mask_heightmap):
    mask_hull = convex_hull_image(target_mask_heightmap)
    if mask_hull[int(best_pix_ind[1]), int(best_pix_ind[2])]:
        return True
    else:
        return False


def get_push_pix(push_maps, num_rotations):
    push_pix_ind = np.unravel_index(np.argmax(push_maps), push_maps.shape)
    push_end_pix_yx = get_push_end_pix_yx(push_pix_ind, num_rotations)
    return push_pix_ind, push_end_pix_yx


def get_push_end_pix_yx(push_pix_ind, num_rotations):
    push_orientation = [1.0, 0.0]
    push_length_pix = 0.1/0.002
    rotation_angle = np.deg2rad(push_pix_ind[0]*(360.0/num_rotations))
    push_direction = np.asarray([push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
                                 push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle)])
    return [push_pix_ind[1] + push_direction[1] * push_length_pix, push_pix_ind[2] + push_direction[0] * push_length_pix]


def check_env_depth_change(prev_depth_heightmap, depth_heightmap, change_threshold=300):
    depth_diff = abs(prev_depth_heightmap-depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.02] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold

    return change_detected, change_value


def check_target_depth_change(prev_depth_heightmap, prev_target_mask_heightmap, depth_heightmap, change_threshold=50):
    prev_mask_hull = binary_dilation(convex_hull_image(prev_target_mask_heightmap), iterations=5)
    depth_diff = prev_mask_hull*(prev_depth_heightmap-depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.02] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold

    return change_detected, change_value


def process_mask_heightmaps(segment_results, seg_mask_heightmaps):
    names = []
    heightmaps = []
    for i in range(len(segment_results['labels'])):
        name = segment_results['labels'][i]
        heightmap = seg_mask_heightmaps[:, :, i]
        if np.sum(heightmap) > 10:
            names.append(name)
            heightmaps.append(heightmap)
    return {'names': names, 'heightmaps': heightmaps}


def get_replay_id(predicted_value_log, label_value_log, reward_value_log, sample_ind, replay_type):
    # Prioritized experience replay, find sample with highest surprise value
    sample_ind = np.asarray(sample_ind)
    predicted_values = np.asarray(predicted_value_log)[sample_ind]
    label_values = np.asarray(label_value_log)[sample_ind]
    reward_values = np.asarray(reward_value_log)[sample_ind]
    if replay_type == 'augment':
        # assume predicted_value for different mask input are close
        label_values = label_values - reward_values + 1.0

    sample_surprise_values = np.abs(predicted_values - label_values)
    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
    sorted_sample_ind = sample_ind[sorted_surprise_ind]
    pow_law_exp = 2
    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
    sample_iteration = sorted_sample_ind[rand_sample_ind]
    print(replay_type.capitalize(), 'replay: iteration %d (surprise value: %f)' %
          (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
    return sample_iteration


def get_pointcloud(color_img, depth_img, masks_imgs, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w-1, im_w), np.linspace(0, im_h-1, im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w, 1)
    cam_pts_y.shape = (im_h*im_w, 1)
    cam_pts_z.shape = (im_h*im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h*im_w, 1)
    rgb_pts_g.shape = (im_h*im_w, 1)
    rgb_pts_b.shape = (im_h*im_w, 1)

    num_masks = masks_imgs.shape[2]
    masks_pts = masks_imgs.copy()
    masks_pts = masks_pts.transpose(2, 0, 1).reshape(num_masks, -1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts, masks_pts


def get_heightmap(depth_img, color_intrin_part, aligned_depth_frame, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base, compensation):
    
    import cv2
    depth_img_cropped = depth_img[15: 463, 95: 543]
    depth_img_resize = cv2.resize(depth_img_cropped, (224, 224))

    idx_proj_mat = np.zeros((224*224, 2), dtype=np.int)
    for i in range(224):
        for j in range(224):
            idx_proj_mat[i*224+j] = [95+j*2, 15+i*2]

    idx_proj_mat = idx_proj_mat.reshape((224, 224, -1))
    depth_heightmap = np.zeros_like(depth_img_resize)
    pointcloud = np.zeros((224*224, 3))
    for i in range(224):
        for j in range(224):
            wld_coor = pix2wld(color_intrin_part, aligned_depth_frame, idx_proj_mat[i, j], R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
            wld_coor[-1] += 0.016 # compensation for desk height ***
            wld_coor[-1] += 0.016 # compensation for desk height ***

            if wld_coor[-1] < -0.1 or wld_coor[-1] > 0.3:
                wld_coor[-1] = 0

            pointcloud[i*224 + j] = wld_coor
            depth_heightmap[i, j] = wld_coor[-1]

            # print('desk height: ', np.mean(pointcloud[-1]))

    return depth_heightmap, pointcloud









# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def gripper2wld(gripper_position, R_gripper, p_g):
    gripper_position, R_gripper, p_g = np.array(gripper_position), np.array(R_gripper), np.array(p_g)

    T = gripper_position[0:3]
    p_world = np.dot(R_gripper, p_g) + T

    return p_world


def wld2gripper(gripper_position, R_gripper, p_w):
    gripper_position, R_gripper, p_c = np.array(gripper_position), np.array(R_gripper), np.array(p_w)
    T = gripper_position

    p_camera = np.dot(np.linalg.inv(R_gripper), p_w - T)

    return p_camera


def pixel2cam(color_intrin_part, aligned_depth_frame, target_pix):

    ppx = color_intrin_part[0]
    ppy = color_intrin_part[1]
    fx = color_intrin_part[2]
    fy = color_intrin_part[3]

    target_depth = aligned_depth_frame.get_distance(target_pix[0], target_pix[1])

    target_xy_true = [(target_pix[0] - ppx) * target_depth / fx,
                      (target_pix[1] - ppy) * target_depth / fy]

    target_cam = np.array([target_xy_true[0] * 1000, target_xy_true[1] * 1000, target_depth * 1000]) # unit: mm

    return target_cam


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


def pix2wld(color_intrin_part, aligned_depth_frame, target_pix, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base):
    
    coor_cam = pix2cam(color_intrin_part, aligned_depth_frame, target_pix)
    coor_gripper = np.dot(R_cam2gripper, coor_cam) + t_cam2gripper
    coor_base = np.dot(R_gripper2base, coor_gripper) + t_gripper2base

    return coor_base


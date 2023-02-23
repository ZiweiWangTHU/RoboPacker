import numpy as np
import sympy
from tqdm import tqdm


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


def ws_pix2wld(color_intrin_part, aligned_depth_frame, target_pix, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base):
    target_pix_prj = [target_pix[0], target_pix[1]]
    coor_cam = pix2cam(color_intrin_part, aligned_depth_frame, target_pix_prj)
    coor_gripper = np.dot(R_cam2gripper, coor_cam) + t_cam2gripper
    coor_base = np.dot(R_gripper2base, coor_gripper) + t_gripper2base

    return coor_base

def point_to(gripper_position, R_gripper, target_position):  # gripper_position, target_position均为世界坐标系下坐标
    gripper_position, R_gripper, target_position = np.asarray(gripper_position), np.asarray(R_gripper), np.asarray(target_position)

    #  YoZ 平面
    p1_tp = [0.0, 0.0, 0.0]
    p2_tp = [0.0, 1.0, 1.0]
    p3_tp = [0.0, 1.0, -1.0]
    #  转换到世界坐标系
    p1_wd = gripper2wld(gripper_position, R_gripper, p1_tp)
    p2_wd = gripper2wld(gripper_position, R_gripper, p2_tp)
    p3_wd = gripper2wld(gripper_position, R_gripper, p3_tp)
    l12 = p1_wd - p2_wd
    l31 = p3_wd - p1_wd
    n_yz = np.cross(l12, l31)  # gripper坐标系下的YoZ平面在世界坐标系下法向量

    pointing_direction = np.asarray([target_position - gripper_position])
    pos_tgt_in_grp = wld2gripper(gripper_position, R_gripper, target_position)  # ***

    alpha = np.arctan(pos_tgt_in_grp[1] / pos_tgt_in_grp[2]) / np.pi * 180
    beta = (np.pi / 2 - np.arccos(
        np.dot(pointing_direction, n_yz) / (np.linalg.norm(pointing_direction) * np.linalg.norm(n_yz))))[0] / np.pi*180

    return alpha, beta


def halt_point(tip_pos, target_pos, dist=0.4):

    x, y, z = sympy.symbols('x y z')
    root1, root2 = np.asarray(sympy.solve([(x-tip_pos[0]) / (tip_pos[0] - target_pos[0]) - (y-tip_pos[1]) / (tip_pos[1] - target_pos[1]),
                                           (x-tip_pos[0]) / (tip_pos[0] - target_pos[0]) - (z-tip_pos[2]) / (tip_pos[2] - target_pos[2]),
                                           (x-target_pos[0])**2 + (y-target_pos[1])**2 + (z-target_pos[2])**2 - dist**2], [x, y, z]), dtype=np.float32)

    dist1 = np.linalg.norm(root1-tip_pos)
    dist2 = np.linalg.norm(root2-tip_pos)

    if dist1 <= dist2:
        halt_pos = root1
    else:
        halt_pos = root2

    return halt_pos


def pre_grasp(target_pos, normal_vector, dist=0.5):
    x, y, z = sympy.symbols('x y z')
    root1, root2 = np.asarray(sympy.solve([(x - target_pos[0]) / normal_vector[0] - (y - target_pos[1]) / normal_vector[1],
                                           (y - target_pos[1]) / normal_vector[1] - (z - target_pos[2]) / normal_vector[2],
                                           (x - target_pos[0])**2 + (y - target_pos[1])**2 + (z - target_pos[2])**2 - dist**2],
                                          [x, y, z]))

    if root1[2] > target_pos[2]:
        point = root1
    else:
        point = root2
    return np.asarray(point, dtype=np.float)


def approximate_plane(bbox, depth_img_normal, color_intrin_part, aligned_depth_frame, target_pix, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base):
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2]
    h = bbox[3]
    x2 = x1 + w
    y2 = y1 + h

    # depthValue, _ = readDepthSensor()
    depthValue = depth_img_normal

    sample_step_x = 5 * w // 8
    sample_step_y = 5 * h // 8
    p_x1 = x1 - ((2*w) // sample_step_x) * sample_step_x
    p_y1 = y1 - ((2*h) // sample_step_y) * sample_step_y
    p_x2 = x2 + ((2*w) // sample_step_x) * sample_step_x
    p_y2 = y2 + ((2*h) // sample_step_y) * sample_step_y

    if p_x1 < 0:
        p_x1 = 0
    if p_x2 > 639:
        p_x2 = 639
    if p_y1 < 0:
        p_y1 = 0
    if p_y2 > 479:
        p_y2 = 479

    samples_plane_x = np.arange(p_x1, p_x2, sample_step_x)
    samples_plane_y = np.arange(p_y1, p_y2, sample_step_y)

    sample_indices = []
    for i in samples_plane_x:
        for j in samples_plane_y:
            sample_indices.append((i, j))

    target_area_x = np.arange(x1, x2, sample_step_x)
    target_area_y = np.arange(y1, y2, sample_step_y)

    target_indices = []
    for i in target_area_x:
        for j in target_area_y:
            target_indices.append((i, j))

    sample_indices = set(sample_indices).difference(set(target_indices))
    sample_indices = [list(item) for item in sample_indices]

    sample_points = []
    for idx in tqdm(sample_indices):
        coordinate = pix2wld(color_intrin_part, aligned_depth_frame, idx, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
        sample_points.append(coordinate)

    sample_points = np.asarray(sample_points)

    A = np.ones_like(sample_points)
    A[:, 0] = sample_points[:, 0]
    A[:, 1] = sample_points[:, 1]
    b = sample_points[:, 2]

    X = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
    normal_vec = np.array([X[0], X[1], -1])

    return normal_vec
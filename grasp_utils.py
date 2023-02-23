import time
import datetime
import cv2
import argparse

import numpy as np
import torch
import pyrealsense2 as rs
from my_utils.rs import get_current_image
from my_utils.rigid_transform import ws_pix2wld
from trainer import Trainer
from logger import Logger
from generator import grasps_generator, get_pointcloud
from generator import push_generator as push_generator_in_grasp
import my_utils.utils as utils
global sample_iteration
# from segmenter import segment
from robot import Robot
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
import torch.backends.cudnn as cudnn
from data import cfg, set_cfg, set_dataset
from layers.output_utils import postprocess, undo_image_transformation
global sample_iteration
import pybullet as p
from my_utils.my_utils import *
from debug_code.axis_unity import *

def process_mask_heightmaps(segment_results, seg_mask_heightmaps, valid_depth_heightmap):
    names = []
    heightmaps = []
    # labels = segment_results['labels'][1:]
    class_ids = segment_results['class_ids']
    for i in range(class_ids.shape[0]):
        name = class_ids[i]
        heightmap = seg_mask_heightmaps[:, :, i]
        # heightmap进行腐蚀
        kernel = np.ones((3, 3), dtype=np.uint8)
        heightmap = cv2.erode(heightmap, kernel, iterations=3)

        # 进一步降噪
        heightmap[valid_depth_heightmap <= 0.002] = 0
        # 原来的阈值为10
        # 现在修改为5
        # print("height_map_mask_num", np.sum(heightmap))
        if np.sum(heightmap) > 10:
            names.append(name)
            heightmaps.append(heightmap)
    return {'class_ids': names, 'heightmaps': heightmaps}

def get_mask_from_heightmap(valid_depth_heightmap):
    heightmaps = np.zeros_like(valid_depth_heightmap)
    heightmaps[valid_depth_heightmap >= 0.001] = 1

    # 因为regrasp只对一个物体进行处理
    return heightmaps.reshape([heightmaps.shape[0], -1, 1])


def grasp_obj_only_depth(robot, pipeline_list, e1h_mtx, R_cam2gripper=None, t_cam2gripper=None, cls_dict=None, target_name=None, use_e1h_camera=True):

    # 仅使用高度图来生成grasp动作
    heightmap_resolution = 0.002

    # 需要进行调整
    workspace_limits = np.asarray([[0.1, 0.7], [-0.42, 0.45], [0.002, 0.4]])
    align = rs.align(rs.stream.color)

    # 根据pipline_list来读取图片
    color_img_list, depth_img_list, aligned_depth_frame_list = get_current_images(pipeline_list, align)
    # R_gripper2base, t_gripper2base = gripper_state(ur5_ip)
    R_gripper2base, t_gripper2base = robot.gripper_state()

    # 但我们只选用top_view的相机来执行grasp
    color_img = color_img_list[-1]
    depth_img = depth_img_list[-1]
    aligned_depth_frame = aligned_depth_frame_list[-1]

    depth_heightmap, point_cloud = get_point_e1h(depth_img, aligned_depth_frame, e1h_mtx,
        R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
    print('mean height', np.mean(point_cloud[:, -1]))
    depth_mean_height = np.mean(depth_heightmap)
    assert (depth_mean_height - np.mean(point_cloud[:, -1]) <= 0.01), "高度图有问题, 请检查"

    masks = np.zeros([1, depth_img.shape[0], depth_img.shape[1]])
    depth_heightmap, seg_mask_heightmaps = get_bullet_heightmap(point_cloud[:, :3], masks, workspace_limits,
        heightmap_resolution=heightmap_resolution)

    # 根据depth_heightmap的高度来决定mask的位置
    # 后面跟高度图后处理
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # 根据regrasp台子的高度进行调整
    regrasp_table_height = 0.2
    seg_mask_heightmaps = np.zeros_like(valid_depth_heightmap)
    seg_mask_heightmaps[valid_depth_heightmap > regrasp_table_height] = 1

    y, x = np.where(seg_mask_heightmaps == 1.0) # only_one_target
    target_center = (int(np.mean(x)), int(np.mean(y)))
    target_position = [target_center[0] * heightmap_resolution + workspace_limits[0][0],
                       target_center[1] * heightmap_resolution + workspace_limits[1][0],
                       valid_depth_heightmap[target_center[1]][target_center[0]] + workspace_limits[2][0]]
    print('target_position', target_position)  # gt

    # 我们需要的是爪子的方向
    rotation_angle, _ = get_axis_from_mask(masks)

    # 此时的rotation_angle已经是与主轴垂直的
    grasp_position = target_position
    grasp_position[-1] -= 0.015  # grasp position is 1cm lower than the target position

    best_rotation_angle = rotation_angle - 90
    target_mask = seg_mask_heightmaps * 255
    curr_pos = robot.get_gripper_pos()
    grasp_succeeded = robot.grasp_depth(curr_pos, grasp_position, best_rotation_angle)
    print('Re-Grasp grasped something?', grasp_succeeded)

    return grasp_succeeded


def garsp_obj(args, robot, net, trainer, device, robot_workspace_limits, pipeline_list, e2h_R_cam2base_list, e2h_t_cam2base_list, e2h_mtx_list, e1h_mtx, R_cam2gripper=None, t_cam2gripper=None, cls_dict=None, target_name=None, use_e1h_camera=True):
    # 配置相关参数
    # heightmap_resolution = args.heightmap_resolution
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    is_testing = True

    load_ckpt = args.load_ckpt  # Load pre-trained ckpt of model
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    heightmap_resolution = 0.002
    workspace_limits = np.asarray([[0.1, 0.7], [-0.42, 0.45], [0.002, 0.4]])

    mode = 'synergic'
    if mode == 'push only':
        conf_thres = 3.0
    elif mode == 'grasp only':
        conf_thres = 0.0
    else:
        conf_thres = 1.0

    print('-----------------------')

    if not is_testing:
        if continue_logging:
            logging_directory = os.path.abspath(args.logging_directory)
            print('Pre-loading data logging session: %s' % logging_directory)
        else:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            logging_directory = os.path.join(os.path.abspath('logs'), timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % logging_directory)
    else:
        logging_directory = os.path.join(os.path.abspath('logs'), 'testing/release')  # ***
        print('Creating data logging session: %s' % logging_directory)

    np.random.seed(random_seed)

    desk_height = 0.0

    init_pos = np.array([0.350, 0.0, 0.55, 2.222, -2.222, 0.0])  # Keep the camera 0.5m directly above the workspace
    align = rs.align(rs.stream.color)  # 设置深度与彩色对齐格式

    if use_e1h_camera:
        init_pos[2] = init_pos[2] + desk_height + t_cam2gripper[2]

    place_pos = np.array([0.2, -0.3, 0.30])
    place_target_pos = np.array([0.59, -0.3, 0.07])
    robot.close()
    print('initializing finish --------------')

    # Initialize trainer
    # trainer = Trainer(0.5, is_testing, load_ckpt, critic_ckpt_file, force_cpu)

    # Define light weight refinenet model
    grasp_fail_count = [0]
    motion_fail_count = [0]
    actions_count_scene = []
    target_name = 'green'

    # 开始进行主循环
    while True:  # main loop
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # 我们直接采集3个相机的图片
        color_img_list, depth_img_list, aligned_depth_frame_list = get_current_images(pipeline_list, align)
        # R_gripper2base, t_gripper2base = gripper_state(ur5_ip)
        R_gripper2base, t_gripper2base = robot.gripper_state()

        # 但我们只选用top_view的相机来执行grasp
        color_img = color_img_list[-1]
        depth_img = depth_img_list[-1]
        aligned_depth_frame = aligned_depth_frame_list[-1]

        # 在线下, 我们可以直接采集到BGR图片格式
        frame = torch.from_numpy(color_img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        with torch.no_grad():
            preds = net(batch)
            # 获得image的大小
            h, w, _ = frame.shape
            t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.9)
        mask = t[3].cpu().numpy()

        class_ids = t[0].cpu().numpy()
        class_ids = class_ids + 1
        # [C, H, W] --> [H, W, C]
        masks = mask.transpose(1, 2, 0)
        segment_results = {'class_ids': class_ids, 'masks': masks}
        if mask.shape[0] <= 0:
            print("all object had been processed, done ------------------------")
            exit()

        seg_mask_heightmaps = masks.copy()

        # 转化为点云
        # 目前的point_cloud与采集的深度图一样
        depth_heightmap, point_cloud = get_point_e1h(depth_img, aligned_depth_frame, e1h_mtx,
            R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)
        print('mean height', np.mean(point_cloud[:, -1]))
        depth_mean_height = np.mean(depth_heightmap)
        assert (depth_mean_height - np.mean(point_cloud[:, -1]) <= 0.01), "高度图有问题, 请检查"

        # 我们必须进一步处理
        # 在分辨率与工作区域的限制下, depth_heightmap and seg_mask_heightmaps 变为[224, 224]
        depth_heightmap, seg_mask_heightmaps = get_bullet_heightmap(point_cloud[:, :3], masks, workspace_limits,
            heightmap_resolution=heightmap_resolution)

        # 后面跟高度图后处理
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # mask_heightmaps [224, 224]
        mask_heightmaps = process_mask_heightmaps(segment_results, seg_mask_heightmaps, valid_depth_heightmap)
        seeking_target = False
        target_mask_heightmap = mask_heightmaps['heightmaps'][0]
        y, x = np.where(target_mask_heightmap == 1.0)
        target_center = (int(np.mean(x)), int(np.mean(y)))
        print('target_center', target_center)
        print('Target name:', mask_heightmaps['class_ids'][0])

        # 使用分辨率得到的target_position结果
        # 我们还是需要分辨率回归到世界坐标系
        target_position = [target_center[0] * heightmap_resolution + workspace_limits[0][0],
                           target_center[1] * heightmap_resolution + workspace_limits[1][0],
                           valid_depth_heightmap[target_center[1]][target_center[0]] + workspace_limits[2][0]]
        print('target_position', target_position)  # gt

        # point_cloud还是可以回来的
        h, w = valid_depth_heightmap.shape
        point_cloud = get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits)
        point_cloud_reshaped = point_cloud.reshape((h, w, -1))
        target_position_1 = point_cloud_reshaped[target_center]
        print("height_map_target_position", target_position_1)

        if not mode == 'push only':
            grasps, grasp_mask_heightmaps, num_grasps = grasps_generator(target_center, point_cloud_reshaped, valid_depth_heightmap)
            # trainer.model.load_state_dict(torch.load('valid_models/grasp/007.pkl'))
            # trainer.model.load_state_dict(torch.load('valid_models/grasp/best_reg_L1.pkl'))
            print('num_grasps', num_grasps)
            if num_grasps == 0:
                primitive_action = 'push'
                target_name = None
                continue
            elif num_grasps > 100:
                sampled_inds = np.random.choice(np.arange(num_grasps), 100, replace=False)
            else:
                sampled_inds = np.random.choice(np.arange(num_grasps), num_grasps, replace=False)
            if num_grasps > 1:
                confs, grasp_inds, rot_inds = [], [], []
                grasp_masks = []
                for i in sampled_inds:
                    grasp_mask_heightmap = grasp_mask_heightmaps[i][0]
                    target_mask = target_mask_heightmap * 255
                    grasp_mask = grasp_mask_heightmap * 255
                    confidence, _ = trainer.forward(valid_depth_heightmap, target_mask, grasp_mask)
                    confs.append(confidence.item())
                    grasp_inds.append(grasp_mask_heightmaps[i][1])
                    rot_inds.append(grasp_mask_heightmaps[i][2])
                    grasp_masks.append(grasp_mask_heightmaps[i][0])

                grasp_inds = np.hstack((np.array(rot_inds).reshape((-1, 1)), np.array(grasp_inds)))
                grasp_masks = np.array(grasp_masks, dtype=np.uint8)
                best_grasp_conf = np.max(confs)
                best_ind = np.argmax(confs)

                print('best_grasp_conf', best_grasp_conf)
                best_grasp_pix_ind = grasp_inds[best_ind][1:3]
                best_grasp_rot_ind = grasp_inds[best_ind][0]
                best_grasp_mask = grasp_masks[best_ind]

            grasp_position = point_cloud_reshaped[best_grasp_pix_ind[1], best_grasp_pix_ind[0]]
            grasp_position[-1] -= 0.015  # grasp position is 1cm lower than the target position
            grasp_position[-1] += desk_height
            best_rotation_angle = 180 / 16 * best_grasp_rot_ind - 90

        ############## Generate and Choose the best push ########################
        pushes = push_generator_in_grasp(target_center, valid_depth_heightmap)
        trainer.model.load_state_dict(torch.load('valid_models/grasp/007.pkl'))
        push_confs = []
        for push in pushes:
            push_start_point = push[0]
            rot_ind = push[1]
            push_mask = push[2]
            push_end_point = push[3]
            confidence, _ = trainer.forward(valid_depth_heightmap, target_mask_heightmap, push_mask)
            push_confs.append(confidence.item())

        print('best push conf: ', np.max(push_confs))
        best_push_ind = np.argmax(push_confs)
        best_push_start_point = pushes[best_push_ind][0]
        best_push_rot_ind = pushes[best_push_ind][1]
        best_push_mask = pushes[best_push_ind][2]
        best_push_end_point = pushes[best_push_ind][3]
        push_start_position = point_cloud_reshaped[best_push_start_point[1], best_push_start_point[0]]
        push_end_position = point_cloud_reshaped[best_push_end_point[1], best_push_end_point[0]]

        push_start_position[-1] += 0.02
        push_end_position[-1] += 0.02
        if seeking_target:  # Seeking target with deterministic pushing
            print('Seeking target')
            primitive_action = 'push'
        if mode == 'grasp only':
            primitive_action = 'grasp'
        elif mode == 'push only':
            primitive_action = 'push'
        else:
            if best_grasp_conf < conf_thres:
                primitive_action = 'push'
                best_pix_ind = [best_push_rot_ind, best_push_start_point[0], best_push_start_point[1]]
            else:
                primitive_action = 'grasp'
                best_pix_ind = best_grasp_pix_ind
        if trainer.iteration == 1:
            primitive_action = 'push'

        ############## Executing action ########################
        motion_fail_count[0] += 1
        if primitive_action == 'push':
            # print('target_position', target_position)
            # print('push_start_position', push_start_position)
            # print('push_end_position', push_end_position)

            robot.push(target_position=target_position, start_position=push_start_position, end_position=push_end_position, curr_pos=robot.get_gripper_pos(), init_pos=init_pos, mode=1)

        if primitive_action == 'grasp':
            grasp_fail_count[0] += 1
            curr_pos = robot.get_gripper_pos()
            # compensation for desk height
            print('grasp_position', grasp_position)

            if robot.grasp_safe(grasp_position):
                grasp_succeeded, target_grasped = robot.grasp(curr_pos, grasp_position, best_rotation_angle, init_pos=init_pos, target_mask=target_mask, grasp_pix_ind=best_grasp_pix_ind)
                print('Grasped something?', grasp_succeeded)

                if grasp_succeeded:
                    print('Target grasped?:', target_grasped)
                    if target_grasped:
                        actions_count_scene.append(motion_fail_count[0])
                        print(actions_count_scene)
                        print('average actions:', np.mean(actions_count_scene))
                        motion_fail_count[0] = 0
                        grasp_fail_count[0] = 0
                        break
                    if target_grasped:
                        robot.place(place_target_pos, init_pos, -best_rotation_angle)

                    else:
                        robot.place(place_pos, init_pos, -best_rotation_angle)  # place object in the box.
                else:
                    print('Grasping failed')
                    grasp_succeeded = False
                    target_grasped = False
                    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=init_pos, angles=-best_rotation_angle)  # move back to observing position
                    robot.open()

        trainer.iteration += 1

    iteration_time_1 = time.time()
    print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

def regrasp_one_obj(args, robot, trainer, device, regrasp_workspace_limits, pipeline_list, e1h_mtx, R_cam2gripper, t_cam2gripper, use_e1h_camera=True):
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    is_testing = True
    mode = 'synergic'
    if mode == 'push only':
        conf_thres = 3.0
    elif mode == 'grasp only':
        conf_thres = 0.0
    else:
        conf_thres = 0.5

    print('-----------------------')
    heightmap_resolution = 0.002
    align = rs.align(rs.stream.color)  # 设置深度与彩色对齐格式
    base_path = os.getcwd()

    init_pos = np.array([0.350, 0.0, 0.55, 2.222, -2.222, 0.0])
    desk_height = 0
    if use_e1h_camera:
        init_pos[2] = init_pos[2] + desk_height + t_cam2gripper[2]

    while True:  # main loop
        # 我们直接采集3个相机的图片
        color_img_list, depth_img_list, aligned_depth_frame_list = get_current_images(pipeline_list, align)
        # R_gripper2base, t_gripper2base = gripper_state(ur5_ip)
        R_gripper2base, t_gripper2base = robot.gripper_state()

        # 但我们只选用top_view的相机来执行grasp
        color_img = color_img_list[-1]
        depth_img = depth_img_list[-1]
        aligned_depth_frame = aligned_depth_frame_list[-1]

        # 目前的point_cloud与采集的深度图一样
        depth_heightmap, point_cloud = get_point_e1h(depth_img, aligned_depth_frame, e1h_mtx,
            R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base)

        print('mean height', np.mean(point_cloud[:, -1]))
        depth_mean_height = np.mean(depth_heightmap)
        print(np.mean(point_cloud[:, -1]))
        assert (abs(depth_mean_height - np.mean(point_cloud[:, -1])) < 0.01), "高度图有问题, 请检查"

        depth_heightmap = get_bullet_heightmap_only(point_cloud[:, :3], regrasp_workspace_limits, heightmap_resolution=heightmap_resolution)
        print(depth_heightmap.shape)

        # 后面跟高度图后处理
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # 手工处理,得到与高度图尺寸一样的mask
        seg_mask_heightmaps = get_mask_from_heightmap(valid_depth_heightmap)

        # 只有一个目标, 故我们不需要处理其他的
        seeking_target = False
        target_mask_heightmap = seg_mask_heightmaps
        y, x = np.where(target_mask_heightmap == 1.0)
        target_center = (int(np.mean(x)), int(np.mean(y)))
        print('target_center', target_center)

        # 使用分辨率得到的target_position结果
        # 我们还是需要分辨率回归到世界坐标系
        target_position = [target_center[0] * heightmap_resolution + regrasp_workspace_limits[0][0],
                           target_center[1] * heightmap_resolution + regrasp_workspace_limits[1][0],
                           valid_depth_heightmap[target_center[1]][target_center[0]] + regrasp_workspace_limits[2][0]]
        print('target_position', target_position)  # gt

        # point_cloud还是可以回来的
        h, w = valid_depth_heightmap.shape
        point_cloud = get_pointcloud(heightmap_resolution, valid_depth_heightmap, regrasp_workspace_limits)
        point_cloud_reshaped = point_cloud.reshape((h, w, -1))
        print(point_cloud_reshaped.shape)
        target_position_1 = point_cloud_reshaped[target_center[1], target_center[0]]
        print("height_map_target_position", target_position_1)



        


def main(args):
    garsp_obj(args)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234)
    parser.add_argument('--force_cpu', dest='force_cpu', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--test_target_seeking', dest='test_target_seeking', action='store_true', default=False)
    parser.add_argument('--max_motion_onecase', dest='max_motion_onecase', type=int, action='store', default=20,
        help='maximum number of motions per test trial')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=5,
        help='number of repeated test trials')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', default=False)
    parser.add_argument('--critic_ckpt', dest='critic_ckpt', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False)
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)











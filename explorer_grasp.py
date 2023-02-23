import argparse
import os
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data import cfg, set_cfg, set_dataset
from yolact import Yolact
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from my_utils.my_utils import *
from robot import Robot
import numpy as np
import chamfer3D.dist_chamfer_3D
import pyrealsense2 as rs
from trainer import Trainer
from push_utils import push_explorer
from grasp_utils import garsp_obj, grasp_obj_only_depth
from regrasp_utils import regrasp_uilts
from pose_utils import get_grasp_point_pose
from place_utils import place_in_box
from models.models.pointnet2_flow import *
from models.models.loss_helper import *
from Planning import PlanModule
from plan_utils import *
from scale_utils import *
from detic.predictor import VisualizationDemo

def main(args):
    pretrain = True

    use_e1h_camera = True
    devices_num = 3
    e2h_camera_num = devices_num
    if use_e1h_camera and devices_num > 1:
        e2h_camera_num = devices_num - 1
    ds5_serial_list = ["141722079365", "048522075245", "140122078690", "048322070276"]

    regrasp_ds5_serial_id = []
    regrasp_pipeline = init_one_cam(regrasp_ds5_serial_id)

    assert (len(ds5_serial_list) >= devices_num), "wrong"


    pipeline_list = init_mulity_cam(devices_num=devices_num, ds5_serial_list=ds5_serial_list)


    base_path = os.getcwd()
    e2h_mtx_list = []
    e2h_R_cam2base_list = []
    e2h_t_cam2base_list = []


    for camera_id in range(e2h_camera_num):
        mtx_path = os.path.join(base_path, "res", "e2h" + "_" + str(camera_id), "mtx.npy")
        mtx = np.load(mtx_path)
        # color_intrin_part = [mtx[0, 2], mtx[1, 2], mtx[0, 0], mtx[1, 1]]  # intrinsics from calibration
        R_cam2base_path = os.path.join(base_path, "res", "e2h" + "_" + str(camera_id), "R_cam2base.npy")
        t_cam2base_path = os.path.join(base_path, "res", "e2h" + "_" + str(camera_id), "t_cam2base.npy")
        R_cam2base = np.squeeze(np.load(R_cam2base_path))
        t_cam2base = np.squeeze(np.load(t_cam2base_path))
        e2h_mtx_list.append(mtx)
        e2h_R_cam2base_list.append(R_cam2base)
        e2h_t_cam2base_list.append(t_cam2base)


    regrasp_mtx_path = os.path.join(base_path, "res", "e2h" + "_" + regrasp_ds5_serial_id, "mtx.npy")
    regrasp_mtx = np.load(regrasp_mtx_path)
    regrasp_R_cam2base_path = os.path.join(base_path, "res", "e2h" + "_" + regrasp_ds5_serial_id, "R_cam2base.npy")
    regrasp_t_cam2base_path = os.path.join(base_path, "res", "e2h" + "_" + regrasp_ds5_serial_id, "t_cam2base.npy")
    regrasp_R_cam2base = np.squeeze(np.load(regrasp_R_cam2base_path))
    regrasp_t_cam2base = np.squeeze(np.load(regrasp_t_cam2base_path))


    if use_e1h_camera:
        e1h_mtx_path = os.path.join(base_path, "res", "e1h", "mtx.npy")
        e1h_mtx = np.load(e1h_mtx_path)
        R_cam2gripper_path = os.path.join(base_path, "res", "e1h", "R_cam2gripper.npy")
        R_cam2gripper = np.squeeze(np.load(R_cam2gripper_path))
        t_cam2gripper_path = os.path.join(base_path, "res", "e1h", "t_cam2gripper.npy")
        t_cam2gripper = np.squeeze(np.load(t_cam2gripper_path))


    robot = Robot()
    robot.close()


    model_base_path = os.getcwd()
    if pretrain:
        print("use pretrain model")
        net = VisualizationDemo(cfg, args)
    else:

        model_path = os.path.join(model_base_path, "weights", "yolact_resnet50_11111_100000.pth")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trained_model = model_path
        config_name = "yolact_resnet50_config_my"
        set_cfg(config_name)
        with torch.no_grad():
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            dataset = None
            net = Yolact()
            net = net.to(device)
            net.load_weights(trained_model)
            net.eval()


    cls_dict_path = os.path.join(model_base_path, "data", "cls_dict.npy")
    cls_dict = np.load(cls_dict_path, allow_pickle=True).item()


    model_path = os.path.join(model_base_path, "weights", "yolact_resnet50.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model = model_path
    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset = None
        net_seg_pose = Yolact()
        net_seg_pose = net_seg_pose.to(device)
        net_seg_pose.load_weights(trained_model)
        net_seg_pose.eval()


    is_testing = True
    load_ckpt = args.load_ckpt  # Load pre-trained ckpt of model
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    heightmap_resolution = 0.002
    workspace_limits = np.asarray([[0.1, 0.7], [-0.42, 0.45], [0.002, 0.4]])
    force_cpu = args.force_cpu
    # Initialize trainer
    trainer = Trainer(0.5, is_testing, load_ckpt, critic_ckpt_file, force_cpu)
    trainer.model.load_state_dict(torch.load('valid_models/grasp/007.pkl'))
    # trainer.eval()


    Tes = "./fsnet/Tes.pth"
    shape_model_path = "./shapenet/pts_flow.pth"
    shape_model = DeformFlowNet(additional_channel=0)
    shape_model.load_state_dict(torch.load(shape_model_path))
    shape_model.cuda()
    shape_model.eval()

    classifier_ce = Point_center_res_cate()  ## scale_model
    classifier_Rot_red = Rot_red(F=1296, k=6)  ## rotation red
    classifier_Rot_green = Rot_green(F=1296, k=6)  ### rotation green
    with torch.no_grad():
        classifier_ce = nn.DataParallel(classifier_ce)
        classifier_ce = classifier_ce.eval()
        classifier_ce.to(device)
        classifier_ce.load_state_dict(torch.load(Tes))



    resolution = 200
    TopHeight = 0.3
    PM = PlanModule(resolution, TopHeight, True)
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    while 1:
        while 1:
            obj_pts_list_all, obj_class_all, tmp_list, box_pts = push_explorer(robot, net, device, workspace_limits, pipeline_list, e2h_R_cam2base_list,
                          e2h_t_cam2base_list, e2h_mtx_list, e1h_mtx, chamLoss, R_cam2gripper, t_cam2gripper, use_e1h_camera=True)

            scale_list, scale_pts, scale_uncertainty_list = get_scale(obj_pts_list_all, obj_class_all)

            # 规划模块规划顺序
            Hts, Hbs = load_heightmap(tmp_list, scale_list)
            Hc = get_height_map_box(box_pts)
            order = PM.sequence(obj_class_all, tmp_list, scale_uncertainty_list, Hts, Hc)
            # 在Unpacked中的编号
            item_idx = order[0]
            x, y, x_center, y_center, z, euler = PM.placement(tmp_list[item_idx], Hts[item_idx], Hbs[item_idx], Hc)
            if scale_uncertainty_list[order[1]] > 0.3:
                continue
            else:
                break
        if len(obj_class_all) == 0:
            break


        garsp_obj(args, robot, net, trainer, device, workspace_limits, pipeline_list, e2h_R_cam2base_list,
                  e2h_t_cam2base_list, e2h_mtx_list, e1h_mtx, R_cam2gripper, t_cam2gripper, cls_dict, target_name=obj_class_all[order[0]], use_e1h_camera=True)

        pred_pose = get_grasp_point_pose(workspace_limits, regrasp_pipeline, net_seg_pose, device, regrasp_R_cam2base,
                                    regrasp_t_cam2base)

        target_pose = [x, y, x_center, y_center, z, euler]  # 规划位姿
        plan_place_pose, box_min, box_max = np.asarray([x_center, y_center]), np.asarray([-0.12, 0.16]), np.asarray([0.24, 0.52])  # 规划位姿，盒子大小

        obj_pose = pred_pose  # 位姿估计结果，在世界坐标系下
        target_name = obj_class_all[order[0]]  # 分割结果，物体类别
        regrasp_init_pos = np.asarray([0.09, 0.324])  # 4D 待抓取位姿

        regrasp_uilts(target_pose, obj_pose, target_name, robot, regrasp_init_pos, points=None)

        regrasp_workspace_limits = np.asarray([[0.1, 0.7], [-0.42, 0.45], [0.002, 0.4]])
        #garsp_obj(args, robot, net, trainer, device, regrasp_workspace_limits, pipeline_list, e2h_R_cam2base_list,
        #    e2h_t_cam2base_list, e2h_mtx_list, e1h_mtx, R_cam2gripper, t_cam2gripper, use_e1h_camera=True)
        grasp_succeeded = grasp_obj_only_depth(robot, pipeline_list, e1h_mtx, R_cam2gripper, t_cam2gripper, use_e1h_camera=True)


        place_init_pose = [0.091, 0.35, 0.517]
        if not grasp_succeeded:
            robot.robot_only_move(robot.gripper_state(), regrasp_init_pos)
        else:
            place_in_box(robot, plan_place_pose, box_min, box_max)




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
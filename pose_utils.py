import os
from realsense_in_robot.my_utils import icp
from my_utils.my_utils import *
import numpy as np
import torch
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
import torch.backends.cudnn as cudnn
from data import cfg, set_cfg, set_dataset
from layers.output_utils import postprocess, undo_image_transformation
from uti_tool import load_ply, draw_cors_withsize, draw_cors, get_3D_corner, trans_3d, gettrans, get6dpose1

def get_grasp_point_pose(workspace_limits, pipeline, net, device, e2h_R_cam2base, e2h_t_cam2base, e2h_mtx, obj_class, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red):


    model_sizes = np.array([[120, 171, 39], [138, 129, 39], [82, 48, 67], [60, 58, 97], [346, 200, 335], [93, 74, 65],
                            [35, 82, 107], [48, 33, 96], [92, 94, 29], [93, 74, 65], [25, 47, 88]])  ## 6x3

    name_list = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]


    align = rs.align(rs.stream.color)
    color_img, depth_img, aligned_depth_frame = get_one_image(pipeline, align=align)


    position, depth_heightmap = get_points(depth_img, aligned_depth_frame, e2h_mtx, e2h_R_cam2base, e2h_t_cam2base)


    frame = torch.from_numpy(color_img).to(device).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    with torch.no_grad():
        preds = net(batch)

        h, w, _ = frame.shape
        t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.9)

    mask = t[3].clone()
    gripper_mask = mask.cpu().numpy()

    gripper_mask = gripper_mask.reshape([-1, 1])
    pts = np.concatenate((position, gripper_mask), axis=1)
    gripper_pts = pts[pts[:, 3] == 1]


    center = np.mean(gripper_pts, axis=0)
    r = 0.15
    valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(
        pts[:, 0] >= center[0]-0.15, pts[:, 0] < center[0]+0.15),
        pts[:, 1] >= center[1]-0.15), pts[:, 1] < center[1]-0.15),
        pts[:, 2] >= center[2]-0.15), pts[:, 2] < center[2]+0.1)

    obj_pts = pts[valid_ind]
    obj_pts = obj_pts[obj_pts[:, 3] != 1]
    obj_pts = obj_pts[:, :3]



    tmp_base_path = "./fsnet/data/tmp"
    tmp_path = os.path.join(tmp_base_path, obj_class + ".txt")
    tmp_pose_path = os.path.join(tmp_base_path, obj_class + "_pose" + ".txt")
    tmp_pose = np.loadtxt(tmp_pose_path)
    obj_tmp = np.load(tmp_path)
    # pc = obj_tmp * 1000.0

    obj_class_id = name_list.index(obj_class)
    model_size = model_sizes[obj_class_id]

    pred_R = icp_process(obj_pts, obj_tmp, tmp_pose, device)
    """
    
    OR, x_r, y_r, z_r = get_3D_corner(pc)
    points = torch.from_numpy(obj_pts.astype(np.float32)).unsqueeze(0)
    ptsori = points.clone()
    points = points.numpy().copy()
    res = np.mean(points[0], 0)
    points[0, :, 0:3] = points[0, :, 0:3] - np.array([res[0], res[1], res[2]])
    points = torch.from_numpy(points).to(device)

    pointsf = points[:, :, 0:3].unsqueeze(2)
    points = pointsf.transpose(3, 1)
    points_n = pointsf.squeeze(2)
    cate_id = torch.zeros((1, 1))
    cate_id[0][0] = obj_class_id + 1
    one_hot = torch.zeros(points.shape[0], 16).scatter_(1, cate_id.cpu().long(), 1)
    one_hot = one_hot.cuda()

    # model = pc[np.random.choice(len(pc), 500, replace=True), :]
    # model = torch.from_numpy(model).unsqueeze(0).cuda()
    # points_n = torch.cat([points_n, model], dim=1)

    pred_seg, point_recon, feavecs = classifier_seg3D(points_n, one_hot)
    pred_choice = pred_seg.data.max(2)[1]

    p = pred_choice
    ptsori = ptsori.to(device)
    pts_ = torch.index_select(ptsori[0, :, 0:3], 0, p[0, :].nonzero()[:, 0])
    feat = torch.index_select(feavecs[0, :, :], 0, p[0, :].nonzero()[:, 0])
    pts_s = pts_[:, :].unsqueeze(0).float()

    corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]]))
    pts_s = pts_s.to(device)
    feat = feat.to(device)
    corners0 = corners0.to(device)

    pts_s = pts_s.transpose(2, 1)

    cen_pred, obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), torch.Tensor([obj_class_id]))
    T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)  ## 1x3x1

    feavec = feat.unsqueeze(0).transpose(1, 2)
    kp_m = classifier_Rot_green(feavec)
    corners_ = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])

   
    # bbx_3D = model_size + obj_size.detach().cpu().numpy()
    # model_3D = np.array([x_r, y_r, z_r])

    box_pred_gan = classifier_Rot_red(feat.unsqueeze(0).transpose(1, 2))

    pred_axis = np.zeros((3, 3)) 
    pred_axis[0:2, :] = kp_m.view((2, 3)).detach().cpu().numpy()

    pred_axis[2, :] = box_pred_gan.view((2, 3)).detach().cpu().numpy()[1, :]

    box_pred_gan = box_pred_gan.detach().cpu().numpy()
    box_pred_gan = box_pred_gan / np.linalg.norm(box_pred_gan)
    cor0 = corners0.cpu().numpy()
    cor0 = cor0 / np.linalg.norm(cor0)

    pred_axis = pred_axis / np.linalg.norm(pred_axis)
    pose = gettrans(cor0.reshape((3, 3)), pred_axis.reshape((3, 1, 3)))
    R = pose[0][0:3, 0:3]
    T = (pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)).view(1, 3).detach().cpu().numpy()
    torch.cuda.empty_cache()
    """
    return pred_R


def icp_process(obj_pts, tmp_pts, tmp_pose, device):
    # obj_pts: [n, 3]
    # tmp_pts: [m, 3]
    # tmp_pose: [4, 4]


    T, distances, iterations = icp.icp(tmp_pts, obj_pts, tolerance=0.000001, sameShape=False)


    C = np.ones((tmp_pts.shape[0], 4))
    C[:, 0:3] = np.copy(tmp_pts)

    C = np.dot(T, C.T).T

    pred_T = T @ tmp_pose
    pred_R = pred_T[:3, :3]

    return pred_R


























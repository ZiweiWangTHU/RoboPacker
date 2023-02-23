# @Time    : 10/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import argparse
import pickle
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov3_fsnet.models.experimental import attempt_load
from yolov3_fsnet.utils.datasets import LoadStreams, LoadImages, LoadImages_fsnet
from yolov3_fsnet.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov3_fsnet.utils.plots import plot_one_box
from yolov3_fsnet.utils.torch_utils import select_device, load_classifier, time_synchronized
from uti_tool import getFiles_ab_cate, depth_2_mesh_bbx, load_ply, show_3D_single, loss_recon
from Net_deploy import load_models, FS_Net_Test
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def detect(opt, data_path, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red,
           model_size, cate_id0, cate):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride = int(model.stride.max())  # model stride
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # if half:
    #     model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages_fsnet(data_path, img_size=imgsz, stride=stride)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    iou, R_l, T_l, rec_l, seg_l, rec_seg_l, l_T = [], [], [], [], [], [], []

    for icc, data in enumerate(dataloader):
        path, img, im0s, depth_, Rt, Tt, seg, box, pc, pts_rec, seg, pts_seg, cen_gt = data

        img = img[0].to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        # pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # pred, cenxy = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
        #                                   agnostic=opt.agnostic_nms)
        # add pred and cenxy
        # torch.save(pred, 'pred.pt')
        # torch.save(cenxy, 'cenxy.pt')
        # pred = torch.load('pred.pt')
        # cenxy = torch.load('cenxy.pt')

        # pred2 = pred[0][(np.where(pred[0][:,-1].cpu()==63))] ##laptop
        K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        # DR = int(cenxy.cpu().numpy()[1])
        # DC = int(cenxy.cpu().numpy()[0])
        depth = depth_[0].numpy()
        # if depth[DR, DC] == 0:
        #     while depth[DR, DC] == 0:
        #         DR = min(max(0, DR + np.random.randint(-10, 10)), 480)
        #         DC = min(max(0, DC + np.random.randint(-10, 10)), 640)
        # XC = [0, 0]
        # XC[0] = np.float32(DC - K[0, 2]) * np.float32(depth[DR, DC] / K[0, 0])
        # XC[1] = np.float32(DR - K[1, 2]) * np.float32(depth[DR, DC] / K[1, 1])
        # cen_depth = np.zeros((1, 3))
        # cen_depth[0, 0:3] = [XC[0], XC[1], depth[DR, DC]]

        # Process detections
        for i in range(1):  # detections per image
            det = [1, 2]
            p, s, im0 = path[0], '', im0s[0].numpy()
            mode = 'image'
            p = Path(p)  # to Path

            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                #     Rescale boxes from img_size to im0 size
                #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                # label = f'{names[int(cls)]} {conf:.2f}'

                # name = data[0][0].split('/')[-1].split('_')[0]
                name = data[0][0][:len(data[0][0]) - 7] + 'label.pkl'
                ground = pickle.load(open(name, 'rb'))
                # for ind, name in enumerate(ground['model_list']):
                #     if name.find(cate) != -1:
                #         idx = ind



                # get bbox

                bbbox = ground['bboxes']
                # plot_one_box([bbbox[1], bbbox[0], bbbox[3], bbbox[2]], im0, label='', color=[0, 0, 255],
                #              line_thickness=3)
                dep3d = depth_2_mesh_bbx(depth, [bbbox[0], bbbox[2], bbbox[1], bbbox[3]], K)
                # dep3d = depth_2_mesh_bbx(depth, [det[0][1],det[0][3],det[0][0],det[0][2]], K)
                dep3d = dep3d[np.where(dep3d[:, 2] > 0.0)]
                # show_mulit_mesh([dep3d])
                cen_depth = np.array([-100, -180, 500]).reshape((1, 3))
                dep3d = chooselimt_test(dep3d, 400, cen_depth)  ##3 *N
                choice = np.random.choice(len(dep3d), 2000, replace=True)
                dep3d = dep3d[choice, :]
                show_3D_single(dep3d, 'ori')
                # dep =depth
                # seg = seg[0].numpy()
                # for x in range(480):
                #     for y in range(640):
                #         if seg[x][y] != 255:
                #             dep[x][y] = 0
                # dep3d_seg = depth_2_mesh_bbx(dep, [bbbox[0], bbbox[2], bbbox[1], bbbox[3]], K)
                # # dep3d = depth_2_mesh_bbx(depth, [det[0][1],det[0][3],det[0][0],det[0][2]], K)
                # dep3d_seg = dep3d_seg[np.where(dep3d[:, 2] > 0.0)]
                # # show_mulit_mesh([dep3d])
                # cen_depth = np.array([-100, -180, 500]).reshape((1, 3))
                # dep3d_seg = chooselimt_test(dep3d_seg, 400, cen_depth)  ##3 *N
                # choice = np.random.choice(len(dep3d_seg), 3000, replace=True)
                # dep3d_seg = dep3d_seg[choice, :]

                iou3d, R_loss, T_loss, Rec_loss, Seg_loss, pts_s1, rec_seg_loss, loss_T = FS_Net_Test(dep3d, pc[0].numpy(), im0,
                                                                                              Rt, Tt, classifier_seg3D,
                                                                                              classifier_ce,
                                                                                              classifier_Rot_green,
                                                                                              classifier_Rot_red,
                                                                                              cate, model_size,
                                                                                              cate_id0, seg, pts_seg,cen_gt,
                                                                                              num_cor=3,
                                                                                              pts_rec=pts_rec)
                # time.sleep(100)

                choice = np.random.choice(2000, len(pts_s1), replace=True)
                pts_seg = pts_seg[:, choice, :]
                pts_s1 = torch.tensor(pts_s1).unsqueeze(0).cuda()
                # loss_seg=loss_recon(pts_s1,pts_seg)
                pts_s1 = pts_s1.cuda()
                pts_seg = pts_seg.cuda()
                loss_seg = loss_recon(pts_s1, pts_seg)
                print('         IoU:%f, R_loss:%f,  T_loss:%f,  Recon_loss:%f,  Seg_loss:%f' % (
                    iou3d, R_loss, T_loss, Rec_loss, loss_seg.item()))
                iou.append(iou3d)
                R_l.append(R_loss)
                T_l.append(T_loss)
                rec_l.append(Rec_loss)
                seg_l.append(loss_seg.item())
                rec_seg_l.append(rec_seg_loss)
                l_T.append(loss_T)

                print(icc)

    iou50, iou75, d5cm5, d10cm5 = 0, 0, 0, 0
    for i in range(len(dataloader)):
        if iou[i] > 0.5:
            iou50 += 1
            if iou[i] > 0.75:
                iou75 += 1
        if T_l[i] < 50:
            if R_l[i] < 10:
                d10cm5 += 1
                if R_l[i] < 5:
                    d5cm5 += 1
    iou75 = iou75 / len(dataloader)
    iou50 = iou50 / len(dataloader)
    d5cm5 = d5cm5 / len(dataloader)
    d10cm5 = d10cm5 / len(dataloader)
    iou_aver = np.mean(iou)
    R_aver = np.mean(R_l)
    T_aver = np.mean(T_l)
    eva = ['IoU:%4f' % (iou_aver * 100), 'IoU75:%4f' % (iou75 * 100), 'IoU50:%4f' % (iou50 * 100),
           '5d5cm:%4f' % (d5cm5 * 100),
           '10d5cm:%4f' % (d10cm5 * 100), 'R_loss:%4f' % R_aver, 'T_loss:%4f' % T_aver, 'Rec_loss:%4f' % np.mean(rec_l),
           'Seg_loss:%4f' % np.mean(seg_l), 'Rec_seg_loss:%4f' % np.mean(rec_seg_l),'loss_T:%4f' % np.mean(l_T)]
    eval_file = open('%s_eval_logs.txt' % cate, 'w')
    for ev in eva:
        print(ev)
        eval_file.write(ev + '\n')
    eval_file.close()


def chooselimt_test(pts0, dia, cen):  ##replace the 3D sphere with 3D cube

    pts = pts0.copy()
    pts = pts[np.where(pts[:, 2] > 20)[0], :]
    ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia)[0], :]
    if ptsn.shape[0] < 1000:
        ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 2)[0], :]
        if ptsn.shape[0] < 500:
            ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 3)[0], :]
    return ptsn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default=63, nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default='False', action='store_true', help='existing project/name ok, '
                                                                                 'do not increment')
    opt = parser.parse_args()
    print(opt)

    # cate = 'cracker_box'
    cate = 'power_drill'
    # cate = 'mustard_bottle'
    # cate = 'mug_deformed'
    # cate = 'camera'
    # fold = '/home/charlielee/FS_Net-main/yolov3_fsnet/data/test_scene_1/' ##should be absolute path
    fold = '/home/lcl/fsnet_oper/yolov3_fsnet/data/%s_deform/' % cate
    fold = '/home/lcl/FS_Net-main/yolov3_fsnet/data/%s_deform1/' % cate
    # fold = '/home/lcl/fsnet_test/yolov3_fsnet/data/%s/' % cate
    # fold = '/home/lcl/display_5/%s/' % cate
    # fold = '/home/lcl/grasp_pybullet/dataset/%s/1/' % cate

    classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0 = load_models(
        cate)
    with torch.no_grad():
        detect(opt, fold, classifier_seg3D, classifier_ce, classifier_Rot_green,
               classifier_Rot_red, model_size, cate_id0, cate)

import pybullet as p
import math
from collections import namedtuple
import numpy as np
import os
import pybullet_data as pd
import random
import time
import cv2
import _pickle as cPickle
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R


from my_utils.my_utils import *
from robot import Robot
from my_utils.rotations import *

def gripper_pos2eur(pos):

    gripper_rvec = pos[3: 6]
    gripper_euler = rotVec2Euler(gripper_rvec)

    return gripper_euler

def get_mat_from_eur(eur):
    Quaternion = p.getQuaternionFromEuler(eur)
    mat = np.asarray(p.getMatrixFromQuaternion(Quaternion)).reshape((3, 3))
    return mat

def get_eur_from_mat(rot_mat):
    rot_mat = rot_mat[:3, :3].copy()
    r3 = R.from_matrix(rot_mat)
    qua = r3.as_quat()

    targer_eur = p.getEulerFromQuaternion(qua)
    return targer_eur

def trans_pos_mat(robot, targer_mat, obj_mat):

    trans_eur_list = [[np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi],
                      [np.pi, np.pi, 0], [np.pi, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]

    trans_eur_list = np.asarray(trans_eur_list)
    # pos = p.getLinkState(self.id, self.eef_id)
    # UR5_rot = pos[5]
    pos = robot.get_gripper_pos()
    gripper_eur = gripper_pos2eur(pos)
    UR5_mat = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(gripper_eur))).reshape([3, 3])
    # UR5_mat = np.asarray(pos).reshape((3, 3))
    UR5_eur_list = []
    trans_target_mat_list = []

    while len(trans_target_mat_list) == 0:
        for i in range(len(trans_eur_list)):


            trans_eur_one = trans_eur_list[i]
            trans_mat = get_mat_from_eur(trans_eur_one)


            targer_mat_one = trans_mat @ targer_mat
            T = targer_mat_one @ np.linalg.inv(obj_mat)


            UR5_eur = get_eur_from_mat(T @ UR5_mat)
            print(UR5_eur)

            # if UR5_eur[0] >= 0 and UR5_eur[1] >= 0 and UR5_eur[2] >= 0:
            if UR5_eur[1] < 0:

                UR5_eur_list.append(UR5_eur)
                trans_target_mat_list.append(targer_mat_one)


    UR5_eur_pitch_list = [abs(x[2]) for x in UR5_eur_list]
    min_id = UR5_eur_pitch_list.index(min(UR5_eur_pitch_list))
    print(UR5_eur_list[min_id])

    return trans_target_mat_list[min_id]


def pose_mat2post_and_rot(pose_mat):

    place = pose_mat[:3, 3]
    rot_mat = pose_mat[:3, :3]
    r3 = R.from_matrix(rot_mat)
    qua = r3.as_quat()
    targer_eur = p.getEulerFromQuaternion(qua)

    return place, targer_eur

# target_pose [4×4] --> obj_pose

def get_regrasp_place_height(points, trans_mat, table_height = 0.3):
    # points为位姿估计中,分割的结果
    new_points = trans_mat @ points
    min_z = np.min(new_points[:, 2])
    place_height = table_height + min_z
    return place_height



def regrasp_uilts(target_pose, obj_pose, target_name, robot, init_pos, points=None):

    sym_obj = ["003_cracker_box.urdf", "007_tuna_fish_can.urdf"]

    target_place = target_pose[:3, 3]

    # target_final_place = [target_place[0] - 0.6, target_place[1] + 1.0, target_place[2]]


    obj_r = target_pose[:3, :3].copy()
    r3 = R.from_matrix(obj_r)
    qua = r3.as_quat()


    targer_eur = p.getEulerFromQuaternion(qua)
    obj_place, obj_rot = pose_mat2post_and_rot(obj_pose)


    if target_name in sym_obj:
        trans_target_mat_list = trans_pos_mat(target_pose[:3, :3], obj_rot, robot)
        target_pos_mat = np.asarray(trans_target_mat_list)
        obj_r = target_pos_mat[:3, :3].copy()
        r3 = R.from_matrix(obj_r)
        qua = r3.as_quat()

        targer_eur = p.getEulerFromQuaternion(qua)

    targer_eur_x = targer_eur[0]
    place_eur_x = targer_eur_x
    place_eur_y = targer_eur[1]
    place_eur_z = targer_eur[2]


    place_target_eur = np.asarray([place_eur_x, place_eur_y, place_eur_z])
    place_target_mat = get_mat_from_eur(place_target_eur)

    obj_mat = np.asarray(p.getMatrixFromQuaternion(obj_rot)).reshape((3, 3))
    T = place_target_mat @ np.linalg.inv(obj_mat)


    pos = robot.get_gripper_pos()
    gripper_eur = gripper_pos2eur(pos)
    gripper_mat = get_mat_from_eur(gripper_eur)


    gripper_mat = T @ gripper_mat

    place_height = get_regrasp_place_height(points, T)


    gripper_eur = get_eur_from_mat(gripper_mat)
    gripper_eur = np.asarray(gripper_eur)
    # gripper_qua = p.getQuaternionFromEuler(gripper_eur)

    robot.regarsp_obj(robot.get_gripper_pos(), gripper_eur)

    table_x = 0.2
    table_y = 0.2
    place_pos = np.asarray([table_x, table_y, place_height])
    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=place_pos, angles=0)  # move back to observing position
    robot.open()

    curr_pos = robot.get_gripper_pos()
    curr_rvec = curr_pos[3: 6]
    curr_place = curr_pos[:3]
    new_place = curr_place.copy()
    curr_euler = rotVec2Euler(curr_rvec)
    robot_yaw = abs(curr_euler[2])
    place_1 = min(abs(robot_yaw - 0), abs(robot_yaw - np.pi))
    place_2 = abs(robot_yaw - np.pi / 2)
    if place_1 < place_2:
        new_place[0] = new_place[0] - 0.1
    else:
        new_place[0] = new_place[0] + 0.1
    new_place[2] = new_place[2] + 0.2


    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=new_place, angles=0)  # move back to observing position


    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=init_pos, angles=0)  # move back to observing position


import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from my_utils.rotations import *


def place_in_box(robot, plan_place_pose, box_min, box_max):


    save_threshold = 0.05

    place_init_pose = []
    place_workplace_limits = []
    new_target_final_place, pose_eur, path_k = get_in_box_pos(plan_place_pose, box_min, box_max)
    place_height = new_target_final_place[2] + save_threshold
    curr_pos = robot.get_gripper_pos()
    curr_rvec = curr_pos[3: 6]
    new_pose = np.asarray([new_target_final_place[0], new_target_final_place[1], new_target_final_place[2],
                           curr_rvec[0], curr_rvec[1], curr_rvec[2]])


    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=new_pose, angles=0)  # move back to observing position


    new_place = np.asarray([new_target_final_place[0], new_target_final_place[1], place_height + 0.12])
    robot.open()
    robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=new_pose, angles=0)



    curr_pos = robot.get_gripper_pos()
    k = math.degrees(pose_eur)
    print("final", k)
    if abs(plan_place_pose[0] - new_target_final_place[0]) >= 0.05 or abs(plan_place_pose[1] - new_target_final_place[1]) >= 0.05:
        move_threshold = 0.03
        start_x = plan_place_pose[0] + move_threshold * np.cos(path_k)
        start_y = plan_place_pose[1] + move_threshold * np.sin(path_k)
        new_place = np.asarray([start_x, start_y, curr_pos[2]])
        robot.move_to(curr_pos=robot.get_gripper_pos(), dest_pos=new_place, angles=pose_eur)

        new_place = np.asarray([start_x - 0.04 * np.cos(path_k), start_y - 0.04 * np.sin(path_k), 0.20])
        robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=new_place, angles=pose_eur)

        curr_pos = robot.get_gripper_pos()
        new_place = np.asarray([curr_pos[0], curr_pos[1], 0.50])
        robot.move_to_v(curr_pos=robot.get_gripper_pos(), dest_pos=new_place, angles=pose_eur)



def get_in_box_pos(obj_pose, box_min, box_max):
    x = obj_pose[0]
    y = obj_pose[1]
    z = obj_pose[2]
    wall_min = box_min
    wall_max = box_max

    wall_threshold_min = [wall_min[0]+0.08, wall_min[1]+0.08]
    wall_threshold_max = [wall_max[0] - 0.08, wall_max[1] - 0.08]

    if x < wall_threshold_min[0]:
        x = wall_threshold_min[0]
    if x > wall_threshold_max[0]:
        x = wall_threshold_max[0]
    if y < wall_threshold_min[1]:
        y = wall_threshold_min[1]
    if y > wall_threshold_max[1]:
        y = wall_threshold_max[1]

    new_place = [x, y, z]
    pose_eur, path_k = get_final_pose(obj_pose, new_place)
    return new_place, pose_eur, path_k

def get_final_pose(obj_pose, new_pose):
    x1 = obj_pose[0]
    y1 = obj_pose[1]
    x2 = new_pose[0]
    y2 = new_pose[1]
    k = (y2 - y1)/(x2 - x1)
    pose_k = -(1. / k)
    pose_eur = math.radians(np.arctan(pose_k))
    pose_eur = pose_eur
    path_k = np.arctan(k)

    return pose_eur, path_k

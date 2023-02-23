import socket
import struct
import numpy as np
import time
from my_utils.rotations import euler2Rotmat, rotMat2RotVec, rotVec2RotMat
from my_utils.rigid_transform import point_to
from my_utils.rotations import *
from my_utils.rigid_transform import point_to

def get_state(ip):

    sk = socket.socket()
    sk.connect(ip)

    data = sk.recv(1108)
    dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
           'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
           'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
           'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
           'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
           'Tool Accelerometer values': '3d',
           'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
           'V main': 'd',
           'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
           'Elbow position': 'd', 'Elbow velocity': '3d'}
    names = []
    ii = range(len(dic))
    for key, i in zip(dic, ii):
        fmtsize = struct.calcsize(dic[key])

        # print(fmtsize)
        data1, data = data[0:fmtsize], data[fmtsize:]
        fmt = "!" + dic[key]
        names.append(struct.unpack(fmt, data1))
        dic[key] = dic[key], struct.unpack(fmt, data1)

    return np.array(dic['Tool vector actual'][1])


def gripper_state(ur5_ip):
    gripper_pos = get_state(ur5_ip)
    t_gripper2base = gripper_pos[0: 3]
    rvec_gripper2base = gripper_pos[3: 6]
    R_gripper2base = rotVec2RotMat(rvec_gripper2base)

    return R_gripper2base, t_gripper2base


def move_to(dest_pos, target_pos, R_gripper, sk, angles):
    theta_x, theta_y = point_to(dest_pos, R_gripper, target_pos)
    theta_x = -theta_x
    theta_z = -angles[0]

    #  move to grasp position and grasp
    rotation = np.array([theta_x, theta_y, theta_z])
    # print('rotation', rotation)
    R = euler2Rotmat(rotation)
    R_final = np.dot(R_gripper, R)

    rvec = rotMat2RotVec(R_final)
    desired_pos = [dest_pos[0], dest_pos[1], dest_pos[2], rvec[0], rvec[1], rvec[2]]
    print('desired_pos', desired_pos)
    tcp_command = 'movel(p[%f, %f, %f, %f, %f, %f], a=0.1, v=0.1, t=0, r=0)\n' % (
        desired_pos[0], desired_pos[1], desired_pos[2], desired_pos[3], desired_pos[4], desired_pos[5])
    sk.send(tcp_command.encode('utf8'))
    time.sleep(3)

def move_to_v(curr_pos, dest_pos, sk, angles):
    curr_rvec = curr_pos[3: 6]
    curr_euler = rotVec2Euler(curr_rvec)
    dest_euler = curr_euler.copy()
    dest_euler[-1] -= angles # ***

    dest_rvec = euler2RotVec(dest_euler)

    desired_pos = [dest_pos[0], dest_pos[1], dest_pos[2], dest_rvec[0], dest_rvec[1], dest_rvec[2]]
    # print('desired_pos', desired_pos)
    tcp_command = 'movel(p[%f, %f, %f, %f, %f, %f], a=0.1, v=0.1, t=0, r=0)\n' % (
        desired_pos[0], desired_pos[1], desired_pos[2], desired_pos[3], desired_pos[4], desired_pos[5])
    sk.send(tcp_command.encode('utf8'))
    time.sleep(5)

def grasp_safe(grasp_pos, safe_area=(0.3, 0.65, -0.12, 0.29, -0.185, 0)):

  x, y, z = grasp_pos[0], grasp_pos[1], grasp_pos[2]
  if safe_area[0] < x < safe_area[1] and safe_area[2] < y < safe_area[3] and safe_area[4] < z < safe_area[5]:
    return True
  else:
    return False


def robo_push(target_position, start_position, end_position, curr_pos, sk, mode=1, push_length=0.16):
    if mode == 0:
        # start_position = np.array(target_position)
        end_position = np.array(target_position)
        if target_position[1] < 0:
            start_position[1] -= push_length / 2
            end_position[1] += push_length / 2
        else:
            start_position[1] += push_length / 2
            end_position[1] -= push_length / 2

        # start_position[2] += 0.01
        end_position[2] += start_position[2]
        move_to_v(curr_pos=np.array(curr_pos), dest_pos=start_position, sk=sk, angles=0)
        time.sleep(5)
        start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
        move_to_v(curr_pos=start_position, dest_pos=end_position, sk=sk, angles=0)

    elif mode == 1:
        # start_position = np.array(target_position)
        start_position[2] -= 0.10  # compensate for desk height
        end_position[2] = start_position[2]

        move_to_v(curr_pos=np.array(curr_pos), dest_pos=start_position, sk=sk, angles=0)
        time.sleep(5)
        start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
        move_to_v(curr_pos=start_position, dest_pos=end_position, sk=sk, angles=0)





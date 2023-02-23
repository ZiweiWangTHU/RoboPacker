import socket
import struct
import numpy as np
import time
from my_utils.rotations import *
from my_utils.rigid_transform import point_to


class Robot():


    def __init__(self):
        self.DOport = [6, 7]
        self.ur_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ur_socket.settimeout(10)
        self.ur_socket.connect(('192.168.3.60', 30003))
        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=[0.350, 0.0, 0.45, 2.222, -2.222, 0.0], angles=0)
        time.sleep(1)
        self.open()
    time.sleep(0.2)


    def set_digital_out(self, port, flag=True):
        rg_cmd = """
        sec sceondaryProgram()
            set_standard_digital_out({}, {})
        end
        """.format(port, flag)
        self.ur_socket.send(rg_cmd.encode('utf8'))
        time.sleep(0.1)


    def close(self, lowForce=False):
        self.set_digital_out(self.DOport[1], lowForce)
        self.set_digital_out(self.DOport[0], True)
        time.sleep(1.5)


    def open(self, lowForce=False):
        self.set_digital_out(self.DOport[1], lowForce)
        self.set_digital_out(self.DOport[0], False)
        time.sleep(1.5)


    def check_grasp(self):
        grasped = self.get_digital_input()[-1][0]
        grasped = int(str(int(grasped)), 2)
        return grasped


    def get_state(self):

        sk = socket.socket()
        sk.connect(('192.168.3.60', 30003))
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

            data1, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            names.append(struct.unpack(fmt, data1))
            dic[key] = dic[key], struct.unpack(fmt, data1)

        return dic


    def get_digital_input(self):

        DI = self.get_state()['Digital input bits']
        return DI


    def get_gripper_pos(self):

        gripper_pos = np.asarray(self.get_state()['Tool vector actual'][1])
        return gripper_pos


    def gripper_state(self):
        
        gripper_pos = self.get_gripper_pos()
        t_gripper2base = gripper_pos[0: 3]
        rvec_gripper2base = gripper_pos[3: 6]
        R_gripper2base = rotVec2RotMat(rvec_gripper2base)

        return R_gripper2base, t_gripper2base


    def move_to_v(self, curr_pos, dest_pos, angles):
        velocity = 0.2
        dist = np.linalg.norm((dest_pos[0: 3] - curr_pos[0: 3]))
        # print('distance', dist)
        curr_rvec = curr_pos[3: 6]
        curr_euler = rotVec2Euler(curr_rvec)
        dest_euler = curr_euler.copy()
        dest_euler[-1] -= angles 

        dest_rvec = euler2RotVec(dest_euler)

        desired_pos = [dest_pos[0], dest_pos[1], dest_pos[2], dest_rvec[0], dest_rvec[1], dest_rvec[2]]

        tcp_command = 'movel(p[%f, %f, %f, %f, %f, %f], a=0.1, v=%f, t=0, r=0)\n' % (
            desired_pos[0], desired_pos[1], desired_pos[2], desired_pos[3], desired_pos[4], desired_pos[5], velocity)
        self.ur_socket.send(tcp_command.encode('utf8'))
        time.sleep(dist / velocity * 2)

    def move_to_rotate(self, curr_pos, dest_eur):

        # 执行原地旋转
        velocity = 0.2
        dest_euler = dest_eur.copy()
        dest_rvec = euler2RotVec(dest_euler)
        desired_pos = [curr_pos[0], curr_pos[1], curr_pos[2], dest_rvec[0], dest_rvec[1], dest_rvec[2]]
        tcp_command = 'movel(p[%f, %f, %f, %f, %f, %f], a=0.1, v=%f, t=0, r=0)\n' % (
            desired_pos[0], desired_pos[1], desired_pos[2], desired_pos[3], desired_pos[4], desired_pos[5], velocity)
        self.ur_socket.send(tcp_command.encode('utf8'))
        time.sleep(5)

    def grasp_safe(self, grasp_pos, safe_area=(0.18, 0.65, -0.30, 0.30, -0.05, 0.2)):
        x, y, z = grasp_pos[0], grasp_pos[1], grasp_pos[2]
        if safe_area[0] < x < safe_area[1] and safe_area[2] < y < safe_area[3] and safe_area[4] < z < safe_area[5]:
            return True
        else:
            print('action not safe')
            return False


    def explore(self, start_position, target_position, curr_pos, init_pos=[0.350, 0.0, 0.45, 2.222, -2.222, 0.0], push_length=0.16):
        
        self.close()
        print('robot exploring...')
        end_position = np.array(target_position)
        end_position[-1] += 0.02
        start_position = end_position.copy()
        if target_position[1] < 0:
            start_position[1] -= push_length / 2
            end_position[1] += push_length / 2
        else:
            start_position[1] += push_length / 2
            end_position[1] -= push_length / 2
        
        self.move_to_v(curr_pos=np.array(curr_pos), dest_pos=start_position, angles=0)

        start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
        self.move_to_v(curr_pos=start_position, dest_pos=end_position, angles=0)
        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=0) # move back to initial position
        self.open()
    

    def push(self, target_position, start_position, end_position, curr_pos, mode=1, init_pos=[0.350, 0.0, 0.45, 2.222, -2.222, 0.0], push_length=0.16):
        
        self.close()
        if mode == 0: # random push
            end_position = np.array(target_position)
            if target_position[1] < 0:
                start_position[1] -= push_length / 2
                end_position[1] += push_length / 2
            else:
                start_position[1] += push_length / 2
                end_position[1] -= push_length / 2
            
            print('executing pushing at ', start_position)
            end_position[2] += start_position[2]
            self.move_to_v(curr_pos=np.array(curr_pos), dest_pos=start_position, angles=0)

            start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
            self.move_to_v(curr_pos=start_position, dest_pos=end_position, angles=0)
            self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=0) # move back to initial position
            self.open()

        elif mode == 1: # determinstic push
            # start_position[2] -= 0.10 # compensate for desk height
            end_position[2] = start_position[2]

            print('executing pushing at ', start_position)
            self.move_to_v(curr_pos=curr_pos, dest_pos=start_position, angles=0) # move to start point
            start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
            self.move_to_v(curr_pos=start_position, dest_pos=end_position, angles=0) # push
            time.sleep(1)

            self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=0) # move back to initial position
            self.open()

    def regarsp_obj(self, start_position, end_position, init_pos=[0.350, 0.0, 0.45, 2.222, -2.222, 0.0]):

        # start_position: 机械臂现在的位姿
        # end_position: 调整后位姿
        self.move_to_rotate(curr_pos=start_position, dest_pos=end_position)


    def push_exp(self, target_position, start_position, end_position, curr_pos, mode=1, init_pos=[0.350, 0.0, 0.45, 2.222, -2.222, 0.0], push_length=0.16):

        # 首先关闭机械爪
        self.close()
        end_position[2] = start_position[2]
        print('executing pushing at ', start_position)
        # 先移动到目标上方
        start_position_0 = [start_position[0], start_position[1], 0.2]
        self.move_to_v(curr_pos=np.asarray(curr_pos), dest_pos=start_position_0, angles=0)  # move to start point
        self.move_to_v(curr_pos=np.asarray(curr_pos), dest_pos=start_position, angles=0)  # move to start point
        start_position = np.hstack((start_position, np.array([2.222, -2.222, 0])))
        self.move_to_v(curr_pos=np.asarray(start_position), dest_pos=end_position, angles=0)  # push
        time.sleep(1)

        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=0)  # move back to initial position

    def grasp(self, curr_pos, dest_pos, rot_angles, init_pos, target_mask, grasp_pix_ind):

        self.open() # open gripper
        self.move_to_v(curr_pos, dest_pos, rot_angles) # move to grasp position
        self.close()  # close gripper

        grasp_succeeded = bool(self.check_grasp())
        target_grasped = grasp_succeeded and target_mask[grasp_pix_ind[1], grasp_pix_ind[0]] == 255.0

        return grasp_succeeded, target_grasped

    def grasp_depth(self, curr_pos, dest_pos, rot_angles):
        self.open()  # open gripper
        self.move_to_v(curr_pos, dest_pos, rot_angles)  # move to grasp position
        self.close()  # close gripper

        grasp_succeeded = bool(self.check_grasp())
        return grasp_succeeded

    def place(self, place_pos, init_pos, rot_angle):

        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=self.get_gripper_pos()+np.array([0, 0, 0.2, 0, 0, 0]), angles=0) # move to place position
        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=place_pos, angles=0) # move to place position
        self.open() # open gripper
        self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=rot_angle) # move back to initial position


    # def reposition(init_pos):
    #     self.move_to_v(curr_pos=self.get_gripper_pos(), dest_pos=init_pos, angles=rot_angle)









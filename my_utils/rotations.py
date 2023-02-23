import numpy as np
import math
from cv2 import Rodrigues

def euler2Rotmat(orientation): # copied

    DEG2RAD = 180 / np.pi

    theta_x = orientation[0] / DEG2RAD
    theta_y = orientation[1] / DEG2RAD
    theta_z = orientation[2] / DEG2RAD

    rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]] ,dtype=np.float32)
    ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype=np.float32)
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]], dtype=np.float32)

    R = np.dot(rx, np.dot(ry, rz))

    return R


def RotMat2Euler(R):
    alpha = -math.atan2(R[2, 1], R[2, 2]) / np.pi * 180
    beta = -math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2)) / np.pi * 180
    gamma = -math.atan2(R[1, 0], R[0, 0]) / np.pi * 180

    return np.array([alpha, beta, gamma])


def rotVec2RotMat(rvec):
    R = Rodrigues(rvec)[0]
    return R


def rotMat2RotVec(R):
    rvec = Rodrigues(R)[0]
    return rvec


def rotVec2Euler(rvec):
    R = rotVec2RotMat(rvec)
    euler = RotMat2Euler(R)

    return euler

def euler2RotVec(euler):
    R = euler2Rotmat(euler)
    rvec = rotMat2RotVec(R)
    return rvec
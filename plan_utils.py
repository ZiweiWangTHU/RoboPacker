import numpy as np
import os
import cv2
import pybullet as p
import time


def load_heightmap(templates, scale_list):
    scale_hts_lists = []
    scale_hbs_lists = []
    for i in range(len(templates)):
        Hts = np.load(r'./urdf_tmp/' + templates[i] + '/Ht200.npy',allow_pickle=True)
        Hbs = np.load(r'./urdf_tmp/' + templates[i] + '/Hb200.npy',allow_pickle=True)
        scale_hts_list, scale_hbs_list = change_hm_scale(Hts, Hbs, scale_list[i])
        scale_hts_lists.append(scale_hts_list)
        scale_hbs_lists.append(scale_hts_list)
    return scale_hts_lists, scale_hbs_lists


def change_hm_scale(hts, hbs, scale):
    x_scale, y_scale, z_scale = scale

    scale_hts_list = []
    scale_hbs_list = []
    # 确定修改的顺序
    scale_list = np.asarray([[x_scale, y_scale, z_scale],
                             [x_scale, z_scale, y_scale],
                             [x_scale, y_scale, z_scale],
                             [x_scale, z_scale, y_scale],
                             [z_scale, y_scale, x_scale],
                             [z_scale, y_scale, x_scale]])
    for i in range(len(scale_list)):
        scale_orin = scale_list[i]
        for j in range(4):
            scale_one = scale_orin.copy()
            if j == 0 or j == 2:
                scale_one = scale_one
            else:
                scale_one = np.asarray([scale_one[1], scale_one[0], scale_one[2]])
            # 修改高度图高度
            hm_id = int(i * 4 + j)
            hts_one = hts[hm_id]
            hbs_one = hbs[hm_id]
            hbs_one[np.isinf(hbs_one)] = 1e+5
            h1, w1 = hts_one.shape
            h2, w2 = hbs_one.shape

            h1_scale = int(np.ceil(h1 * scale_one[1]))
            w1_scale = int(np.ceil(w1 * scale_one[0]))
            h2_scale = int(np.ceil(h2 * scale_one[1]))
            w2_scale = int(np.ceil(w2 * scale_one[0]))

            # new_hts_one = np.zeros([h1_scale, w1_scale])
            # new_hbs_one = np.zeros([h2_scale, w2_scale])

            new_hts_one = cv2.resize(hts_one, (w1_scale, h1_scale))
            new_hbs_one = cv2.resize(hbs_one, (w2_scale, h2_scale))
            scale_hts_list.append(new_hts_one * scale_one[2])
            new_hbs_one = new_hbs_one * scale_one[2]
            # new_hbs_one[new_hbs_one > 1e+3] = np.inf
            scale_hbs_list.append(new_hbs_one)

    return scale_hts_list, scale_hbs_list


def GridSearch(files, item, Hc, scale_list):
    Trans = []
    c = 0.001
    Hts = np.load(r'./standard/' + files[item - 5] + '/Ht.npy', allow_pickle=True)
    Hbs = np.load(r'./standard/' + files[item - 5] + '/Hb.npy', allow_pickle=True)

    Hts, Hbs = change_hm_scale(Hts, Hbs, scale=scale_list)
    i = 0
    BoxW, BoxH = 30, 30
    pitch_rolls = np.array([[0, 0], [0, np.pi / 2], [0, np.pi], [0, 3 * np.pi / 2], [np.pi / 2, 0], [3 * np.pi / 2, 0]])
    for pitch_roll in pitch_rolls:
        transforms = np.concatenate((
        np.repeat([pitch_roll], 4, axis=0).T, [np.arange(0, 2 * np.pi, np.pi / 2)]), axis=0).T
        for trans in transforms:
            print(i)
            Ht, Hb = Hts[i], Hbs[i]

            w, h = Ht.shape
            for X in range(0, BoxW - w + 1):
                for Y in range(0, BoxH - h + 1):
                    Z = np.max(Hc[X:X + w, Y:Y + h] - Hb)
                    Update = np.maximum((Ht > 0) * (Ht + Z), Hc[X:X + w, Y:Y + h])
                    # print(np.max(Update) - TopHeight)
                    if np.max(Update) <= 0.3:
                        score = c * (X + Y) + np.sum(Hc) + np.sum(Update) - np.sum(Hc[X:X + w, Y:Y + h])
                        Trans.append(np.array(list(trans) + [X, Y, Z] + [w, h, score]))
            i = i + 1
    Trans = np.array(Trans)
    if len(Trans) != 0:
        Trans = Trans[np.argsort(Trans[:, 8])]
    # print(Trans.shape)
    trans = Trans[0, :]
    # print(trans.shape)
    # exit()
    return trans

def Update_box(item, trans, num):
    move_item(item, trans)

def move_item(item, trans):
    resolution = 200
    xmin, xmax, ymin, ymax = 0., 0.3, 0., 0.3
    _shift = 0.1
    target_euler = trans[0:3]
    target_pos = trans[3:6]
    shift = trans[6:8]
    target_pos[0]/=(resolution/(xmax-xmin))
    target_pos[1]/=(resolution/(ymax-ymin))
    pos, quater = p.getBasePositionAndOrientation(item)
    new_quater = p.getQuaternionFromEuler(target_euler)
    p.resetBasePositionAndOrientation(item, pos, new_quater)
    AABB = p.getAABB(item)
    shift1 = np.array(pos)-(np.array(AABB[0]))
    shift2 = np.array([((xmax-xmin)/resolution)*(0.5)-_shift,((ymax-ymin)/resolution)*(0.5)-_shift,0])
    shift = shift1-shift2
    new_pos = target_pos+shift
    p.resetBasePositionAndOrientation(item, new_pos, new_quater)
    for i in range(100):
        p.stepSimulation()
        time.sleep(1./240.)




def HM_Packing(items, volumes, order, scale_list, Hc):

    Hc = Hc

    print(order)
    U = []
    item_in_box = []
    Trans = np.zeros([len(order), 9])

    for i in range(0, len(order)):

        item = items[order[i]]
        trans = GridSearch(item, Hc, scale_list[order[i]])
        if type(trans) != type(None):
            item_in_box.append(item)
            orientation = trans[0:3]
            X_center = trans[3] + trans[6] / 2
            Y_center = trans[4] + trans[7] / 2
            Z = trans[5]
            print("Pos:%d,%d,%f" % (trans[3], trans[4], trans[5]))
            print("Ori:%f,%f,%f" % (trans[0], trans[1], trans[2]))
            print("Size:%d,%d" % (trans[6], trans[7]))
            Update_box(item, trans, order[i] + 1)
            item_in_box.append(item)
            Hc = Hc
            Trans[order[i], :] = trans
        else:
            U.append(order[i])

    return Trans, U

def get_height_map_box(surface_pts):


    workspace_limits = [[-0.12, 0.16], [0.24, 0.52], [-0.0, 0.3]]
    heightmap_resolution = 0.0014
    heightmap_size = np.floor(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                              (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    sort_z_ind = np.argsort(surface_pts[:, 2])

    surface_pts = surface_pts[sort_z_ind]
    print(surface_pts.shape)
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] > workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] > workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]

    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - 0.0001 - workspace_limits[0][
        0]) / heightmap_resolution).astype(np.int32)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - 0.0001 - workspace_limits[1][
        0]) / heightmap_resolution).astype(np.int32)

    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]

    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    depth_heightmap[np.isnan(depth_heightmap)] = 0

    return depth_heightmap


def create_box_bullet(size, pos):
    size = np.array(size)
    shift = [0, 0, 0]
    color = [1, 1, 1, 1]
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
        rgbaColor=color,
        visualFramePosition=shift,
        halfExtents=size / 2)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
        collisionFramePosition=shift,
        halfExtents=size / 2)
    p.createMultiBody(baseMass=100,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=pos,
        useMaximalCoordinates=True)


def box_image_show(cameraPos, targetPos, tz_vec, fov=50.0,
                   aspect=1.0,
                   nearVal=0.01,
                   farVal=20):
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=800, height=800,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
    )
    return width, height, rgbImg, depthImg, segImg
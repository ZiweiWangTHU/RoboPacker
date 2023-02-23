
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import os
from tqdm import tqdm
from scipy import interpolate
import cv2
from Planning import PlanModule

xmin, xmax, ymin, ymax = 0.,0.3,0.,0.3
resolution = 200
TopHeight = 0.3
N = 20
_shift = 0.1
c = 0.001

def load_items(numbers):
    flags = p.URDF_USE_INERTIA_FROM_FILE
    model_list = []
    item_ids = []
    filenames = []
    for root, dirs, files in os.walk(r'./urdf_tmp/'):
        if dirs != []:
            name_list = dirs
        for file in files:
            if file.find(".urdf")!=-1:
                model_list.append(os.path.join(root,file))            
    for count in range(len(numbers)):
        item_id = p.loadURDF(model_list[numbers[count]], 
                             [(count//5)/4+2.2, (count%5)/4+0.2, 0.1], flags=flags)
        item_ids.append(item_id)
        filenames.append(name_list[numbers[count]])
    return item_ids, filenames
     
def box_hm():
    sep = (xmax-xmin)/resolution
    xpos = np.arange(xmin+sep/2+_shift,xmax+_shift,sep)
    ypos = np.arange(ymin+sep/2+_shift,ymax+_shift,sep)
    HeightMap = np.zeros((resolution, resolution))
    for i in range(len(xpos)):
        xscan, yscan = np.meshgrid(xpos[i],ypos)
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Start = np.insert(ScanArray,2,TopHeight,0).T
        End = np.insert(ScanArray,2,0,0).T
        RayScan = np.array(p.rayTestBatch(Start, End))
        HeightMap[i,:] = (1-RayScan[:,2].astype('float64'))*TopHeight
    if np.max(HeightMap[0,:])>0.2:
        HeightMap[0,:] = 0
    if np.max(HeightMap[:,0])>0.2:
        HeightMap[:,0] = 0
    if np.max(HeightMap[-1,:])>0.2:
        HeightMap[-1,:] = 0
    if np.max(HeightMap[:,-1])>0.2:
        HeightMap[:,-1] = 0
    return HeightMap        

def Update_box(item, trans, num):
    move_item(item, trans)

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
            hm_id = int(i*4 + j)
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
            """
            scale_hts_x = np.around(np.arange(w1) * scale_one[0]).astype(np.int32)
            scale_hts_y = np.around(np.arange(h1) * scale_one[1]).astype(np.int32)
            scale_hbs_x = np.around(np.arange(w2) * scale_one[0]).astype(np.int32)
            scale_hbs_y = np.around(np.arange(h2) * scale_one[1]).astype(np.int32)
            hts_x = np.arange(w1)
            hts_y = np.arange(h1)
            hbs_x = np.arange(w2)
            hbs_y = np.arange(h2)

            # 进行赋值
            m_hts_x, m_hts_y = np.meshgrid(scale_hts_x, scale_hts_y)
            m_hbs_x, m_hbs_y = np.meshgrid(scale_hbs_x, scale_hbs_y)
            m_x_1, m_y_1 = np.meshgrid(hts_x, hts_y)
            m_x_2, m_y_2 = np.meshgrid(hbs_x, hbs_y)
            print(np.unique())
            new_hts_one[m_hts_y, m_hts_x] = hts_one[m_y_1, m_x_1] * scale_one[2]
            new_hbs_one[m_hbs_y, m_hbs_x] = hbs_one[m_y_2, m_x_2] * scale_one[2]

            # 开始插值
            
            X = np.arange(w1_scale)
            Y = np.arange(h1_scale)
            X1, Y1 = np.meshgrid(X, Y)
            Z = hts_one * scale_one[2]

            print(Z.shape)
            f1 = interpolate.interp2d(m_hts_x, m_hts_y, Z, kind='linear')
            new_hts_one = f1(X, Y)

            X = np.arange(w2_scale)
            Y = np.arange(h2_scale)
            X1, Y1 = np.meshgrid(X, Y)
            Z = hbs_one * scale_one[2]
            f1 = interpolate.interp2d(m_hbs_x, m_hbs_y, Z, kind='linear')
            new_hbs_one = f1(X, Y)

            """
            scale_hts_list.append(new_hts_one * scale_one[2])
            new_hbs_one = new_hbs_one * scale_one[2]
            # new_hbs_one[new_hbs_one > 1e+3] = np.inf
            scale_hbs_list.append(new_hbs_one)

    return scale_hts_list, scale_hbs_list


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

def GridSearch(item, Hc, scale_list):
    Trans = []
    Hts = np.load(r'./standard/' + files[item-5] + '/Ht.npy',allow_pickle=True)
    Hbs = np.load(r'./standard/' + files[item-5] + '/Hb.npy',allow_pickle=True)
    """
    for i in tqdm(range(Hts.shape[0])):
        path = "./obj_height_map/hts/" + str(i) + ".png"
        show_hm(Hts[i], path)
    for i in tqdm(range(Hts.shape[0])):
        path = "./obj_height_map/hbs/" + str(i) + ".png"
        show_hm(Hbs[i], path)
    print(Hts[0].shape)
    """
    print(scale_list)
    Hts, Hbs = change_hm_scale(Hts, Hbs, scale=scale_list)
    """
    for i in tqdm(range(len(scale_hbs_list))):
        path = "./obj_height_map/scale_hts/" + str(i) + ".png"
        show_hm(scale_hts_list[i], path)
    for i in tqdm(range(len(scale_hbs_list))):
        path = "./obj_height_map/scale_hbs/" + str(i) + ".png"
        show_hm(scale_hbs_list[i], path)
    """
    i = 0
    BoxW, BoxH = resolution, resolution
    pitch_rolls = np.array([[0, 0],[0, np.pi/2],[0,  np.pi],[0,3*np.pi/2],[np.pi/2, 0],[3*np.pi/2, 0]])
    for pitch_roll in pitch_rolls:
        transforms = np.concatenate((np.repeat([pitch_roll], 4, axis=0).T, [np.arange(0, 2*np.pi, np.pi/2)]),axis=0).T
        for trans in transforms:       
            print(i)
            Ht, Hb = Hts[i], Hbs[i]

            w,h = Ht.shape
            for X in range(0, BoxW-w+1):
                for Y in range(0, BoxH-h+1):
                    Z = np.max(Hc[X:X+w, Y:Y+h]-Hb)
                    Update = np.maximum((Ht>0)*(Ht+Z), Hc[X:X+w,Y:Y+h])
                    # print(np.max(Update) - TopHeight)
                    if np.max(Update) <= TopHeight:
                        score = c*(X+Y)+np.sum(Hc)+np.sum(Update)-np.sum(Hc[X:X+w,Y:Y+h])
                        Trans.append(np.array(list(trans)+[X,Y,Z]+[w,h,score]))
            i = i + 1
    Trans = np.array(Trans)
    if len(Trans)!=0:
        Trans = Trans[np.argsort(Trans[:,8])]
    # print(Trans.shape)
    trans = Trans[0,:]
    # print(trans.shape)
    # exit()
    return trans

def HM_Packing(items, volumes, order, scale_list, Hc):

    # 采集的盒子高度图为[60, 60]
    Hc = box_hm()

    print(order)
    U = []
    item_in_box = []
    Trans = np.zeros([len(order), 9])

    for i in range(0, len(order)):

        item = items[order[i]]
        trans = GridSearch(item, Hc, scale_list[order[i]])
        if type(trans)!=type(None):
            item_in_box.append(item)
            orientation = trans[0:3]
            X_center = trans[3]+trans[6]/2
            Y_center = trans[4]+trans[7]/2
            Z = trans[5]
            print("Pos:%d,%d,%f" % (trans[3],trans[4],trans[5]))
            print("Ori:%f,%f,%f" % (trans[0],trans[1],trans[2]))
            print("Size:%d,%d" % (trans[6],trans[7]))
            Update_box(item, trans, order[i]+1)
            item_in_box.append(item)
            Hc = box_hm()
            Trans[order[i],:] = trans
        else:
            U.append(order[i])

    return Trans, U

def trans_size_pos(size, pos):
    size_ratio = (xmax-xmin)/resolution
    new_size = size*size_ratio
    #pos为左下角xyz位置，换算成中心位置
    mid_pos = pos + size/2
    new_pos = mid_pos*size_ratio + [xmin,ymin,0]
    return new_size, new_pos

def show_hm(hm, path):

    # path = "./obj_height_map/hm.png"
    hm = hm * 1000
    plt.matshow(hm.astype(np.int16), interpolation='nearest')
    # plt.show()
    plt.savefig(path, transparent=True, dpi=800)
    plt.close()


def move_item(item, trans):
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
        
def pack_item(item, pos, euler):
    target_euler = euler
    target_pos = pos
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

def drawAABB(aabb,width=1):
  aabbMin = aabb[0]
  aabbMax = aabb[1]
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 0, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [0, 1, 0], width)
  f = [aabbMin[0], aabbMin[1], aabbMin[2]]
  t = [aabbMin[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [0, 0, 1], width)
  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMin[0], aabbMin[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMin[1], aabbMin[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMin[0], aabbMax[1], aabbMin[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMin[0], aabbMax[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMin[1], aabbMax[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)
  f = [aabbMax[0], aabbMax[1], aabbMax[2]]
  t = [aabbMax[0], aabbMax[1], aabbMin[2]]
  p.addUserDebugLine(f, t, [1, 1, 1], width)

def draw_box(width=5):
    p.addUserDebugLine([xmin,ymin,0],[xmin,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,0],[xmin,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,0],[xmax,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymax,0],[xmax,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,0],[xmax,ymin,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,0],[xmax,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,0],[xmin,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,0],[xmax,ymax,0], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,TopHeight],[xmax,ymin,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymax,TopHeight],[xmax,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmin,ymin,TopHeight],[xmin,ymax,TopHeight], [1, 1, 1], width)
    p.addUserDebugLine([xmax,ymin,TopHeight],[xmax,ymax,TopHeight], [1, 1, 1], width)

def create_box_bullet(size, pos):
    size = np.array(size)
    shift = [0, 0, 0]
    color = [1,1,1,1]
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    rgbaColor=color,
                                    visualFramePosition=shift,
                                    halfExtents = size/2)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                          collisionFramePosition=shift,
                                          halfExtents = size/2)
    p.createMultiBody(baseMass=100,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=collisionShapeId,
                      baseVisualShapeIndex=visualShapeId,
                      basePosition=pos,
                      useMaximalCoordinates=True)

def box_image_show(cameraPos,targetPos,tz_vec,fov=50.0,
        aspect=1.0,
        nearVal=0.01, 
        farVal=20):
    
    viewMatrix = p.computeViewMatrix(
    cameraEyePosition=cameraPos,
    cameraTargetPosition=targetPos,
    cameraUpVector=tz_vec,
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=fov,               # 摄像头的视线夹角
        aspect=aspect,
        nearVal=nearVal,            # 摄像头焦距下限
        farVal=farVal,               # 摄像头能看上限
    )
    
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=800, height=800,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
    )
    return width, height, rgbImg, depthImg, segImg

if __name__ == '__main__':
    if p.getConnectionInfo()['isConnected']:
        p.disconnect()
    physicsClient = p.connect(p.GUI)
    # 取消可视化界面
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    create_box_bullet([xmax-xmin+0.2,0.1,TopHeight],[(xmax-xmin)/2+xmin+_shift,ymin-0.05+_shift,TopHeight/2])
    create_box_bullet([xmax-xmin+0.2,0.1,TopHeight],[(xmax-xmin)/2+xmin+_shift,ymax+0.05+_shift,TopHeight/2])
    create_box_bullet([0.1,ymax-ymin,TopHeight],[xmin-0.05+_shift,(ymax-ymin)/2+ymin+_shift,TopHeight/2])
    create_box_bullet([0.1,ymax-ymin,TopHeight],[xmax+0.05+_shift,(ymax-ymin)/2+ymin+_shift,TopHeight/2])
    item_numbers = np.array(range(8))
    item_ids, files = load_items(item_numbers)
    
    # 输入obj_list
    class_name_list = ['Bowl', 'Pear', 'Box', 'Banana','Can', 'Mug', 'Bottle']
    templates = ['bowl', 'pear', 'box', 'banana','can', 'mug', 'bottle']
    # 对齐item_ids, files
    new_item_ids = []
    for name_one in class_name_list:
        id = files.index(name_one)
        new_item_ids.append(item_ids[id])

    # 获取sclae_list
    # 测试list
    scale_list = np.asarray([[0.8, 0.9, 1],
                             [0.9, 0.8, 0.97],
                             [0.99, 0.98, 1.04],
                             [0.9, 0.97, 1],
                             [0.99, 0.98, 1.04],
                             [0.99, 0.98, 1.04],
                             [0.99, 0.98, 1.04]])
    # 只是可视化, 没有很大的用处
    for item in item_ids:
        AABB = p.getAABB(item)
        drawAABB(AABB)
    Hts, Hbs = load_heightmap(templates, scale_list)
    
    PM = PlanModule(resolution, TopHeight, True)
    uncertainty = [0,0,0,0,0,0,0]
    unpacked = new_item_ids
    while len(unpacked) > 0:
        Hc = box_hm()
        order = PM.sequence(unpacked, templates, uncertainty, Hts, Hc)
        # 在Unpacked中的编号
        item_idx = order[0]
        x, y, x_center, y_center, z, euler = PM.placement(templates[item_idx], Hts[item_idx], Hbs[item_idx], Hc)
        pack_item(unpacked[item_idx], [x, y, z], euler)
        del unpacked[item_idx]
        del templates[item_idx]
        del uncertainty[item_idx]
        del Hts[item_idx]
        del Hbs[item_idx]
    
    '''
    start_time = time.time()
    Trans, U = HM_Packing(new_item_ids, item_volumes, order, scale_list, Hc=[])
    print(time.time() - start_time)
    if len(U)!=0:
        print('item {} are not packed into the box.'.format(U))
    '''
          
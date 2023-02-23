import pyrealsense2 as rs
import numpy as np
import cv2
import glob


def init_cam():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    for i in range(50):
        data = pipeline.wait_for_frames()
        depth = data.get_depth_frame()
        color = data.get_color_frame()

    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()

    # return color, depth, aligned_frames, aligned_depth_frame.
    return pipeline, align


def live_streaming():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    # 深度图像向彩色对齐
    align_to_color = rs.align(rs.stream.color)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET)

            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            images = color_image

            ###################################################
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (4, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)

            cv2.drawChessboardCorners(color_image, (4, 6), corners, ret)
            # cv2.imshow('calib', color_image)
            ###################################################
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
# live_streaming()


def get_pointcloud():

    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe_profile = pipeline.start(config)

    for i in range(100):
        data = pipeline.wait_for_frames()
        # depth = data.get_depth_frame() # original depth data
        # color = data.get_color_frame() # original color data

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    colorful = np.asanyarray(color_frame.get_data())
    colorful = colorful.reshape(-1, 3) # (307200, 3)

    pc.map_to(color_frame)
    points = pc.calculate(aligned_depth_frame)

    vertices = np.asanyarray(points.get_vertices())
    vtx_array = np.zeros((307200, 3))


    for i in range(len(vertices)):
        for j in range(3):
            vtx_array[i][j] = vertices[i][j]

    vtx2 = np.hstack([vtx_array, colorful])

    with open('pointcloud.txt', 'w') as f:
        for i in range(len(vertices)):

            f.write(str(float(vtx2[i][0]))+' '+
                    str(float(vtx2[i][1]))+' '+
                    str(float(vtx2[i][2]))+' '+
                    str(int(vtx2[i][5]))+' '+
                    str(int(vtx2[i][4]))+' '+
                    str(int(vtx2[i][3]))+'\n')

    return vtx2

# pc = get_pointcloud()


def get_image():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    #获取图像，realsense刚启动的时候图像会有一些失真，我们保存第100帧图片。
    for i in range(100):
        data = pipeline.wait_for_frames()
        depth = data.get_depth_frame()
        color = data.get_color_frame()

    #获取内参
    dprofile = depth.get_profile()
    cprofile = color.get_profile()

    cvsprofile = rs.video_stream_profile(cprofile)
    dvsprofile = rs.video_stream_profile(dprofile)

    color_intrin=cvsprofile.get_intrinsics()
    print('color_intrin', color_intrin)
    depth_intrin=dvsprofile.get_intrinsics()
    print('depth_intrin', depth_intrin)
    extrin = dprofile.get_extrinsics_to(cprofile)
    print('extrin', extrin)

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # cv2.imwrite('color.png', color_image)
    # cv2.imwrite('depth.png', depth_image)
    # cv2.imwrite('depth_colorMAP.png', depth_colormap)


def get_current_image(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    # cv2.imwrite('frame_tmp/0/pcd000r.png', color_image)
    # cv2.imwrite('frame_tmp/0/pcd000d.png', depth_image)
    depth_img_normal = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 2.04
    # cv2.imwrite('frame_tmp/0/pcd000d.tiff', depth_img_normal)

    ###########################################################
    pc = rs.pointcloud()
    points = rs.points()

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    colorful = np.asanyarray(color_frame.get_data())
    colorful = colorful.reshape(-1, 3) # (307200, 3)

    pc.map_to(color_frame)
    points = pc.calculate(aligned_depth_frame)

    vertices = np.asanyarray(points.get_vertices())
    vtx_array = np.zeros((307200, 3))


    for i in range(len(vertices)):
        for j in range(3):
            vtx_array[i][j] = vertices[i][j]

    vtx2 = np.hstack([vtx_array, colorful])

    # with open('pointcloud.txt', 'w') as f:
    #     for i in range(len(vertices)):
    #
    #         f.write(str(float(vtx2[i][0]))+' '+
    #                 str(float(vtx2[i][1]))+' '+
    #                 str(float(vtx2[i][2]))+' '+
    #                 str(int(vtx2[i][5]))+' '+
    #                 str(int(vtx2[i][4]))+' '+
    #                 str(int(vtx2[i][3]))+'\n')
    ###########################################################

    return color_image, depth_image / 1000, aligned_frames, aligned_depth_frame, vtx2


def calib_opencv():
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((6 * 4, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob("pics/*.jpg")
    for frame in sorted(images):
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (4, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    return np.asarray(mtx).squeeze(), np.asarray(rvecs).squeeze(), np.asarray(tvecs).squeeze()


def save_pc(pc, name='pc.txt'):

    if pc.shape[-1] == 6:
        with open(name, 'w') as f:
            for i in range(len(pc)):
                f.write(str(float(pc[i][0]))+';'+
                        str(float(pc[i][1]))+';'+
                        str(float(pc[i][2]))+';'+
                        str(int(pc[i][5]))+';'+
                        str(int(pc[i][4]))+';'+
                        str(int(pc[i][3]))+'\n')
    else:
        with open(name, 'w') as f:
            for i in range(len(pc)):
                f.write(str(float(pc[i][0]))+';'+
                        str(float(pc[i][1]))+';'+
                        str(float(pc[i][2]))+'\n')


def visualize_pc(points, skip=5):

    import matplotlib.pyplot as plt

    points = points[:, 0:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
    x = points[point_range, 0]
    y = points[point_range, 1]
    z = points[point_range, 2]
    ax.scatter(x,  # x
               y,  # y
               z,  # z
               c=z,  # height data for color
               cmap='Blues',
               marker=".")
    ax.axis('auto')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('feature visulization')
    ax.axis('off')  # 设置坐标轴不可见
    ax.grid(False)  # 设置背景网格不可见

    plt.show()



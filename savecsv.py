import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import os

pose_name = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
             "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
             "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
             "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
datatemp = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], []]

"""导入模型"""
# 导入solution
mp_pose = mp.solutions.pose
# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
# 导入模型
pose = mp_pose.Pose(static_image_mode=False,  # 静态图片 or 连续帧视频
                    model_complexity=2,  # 人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于二者之间
                    smooth_landmarks=True,  # 是否平滑关键点
                    enable_segmentation=True,  # 是否人体抠图
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)  # 各帧之间的追踪阈值


# 处理单帧的函数
def process_frame(img):
    # BGR转RGB OpenCV以BGR格式（而不是RGB）读取图像
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型来预测结果
    results = pose.process(img_RGB)
    # 若检测出人体关键点
    if results.pose_world_landmarks:
        # 遍历33个骨骼关键点
        for i in range(len(pose_name)):
            # 获取关键点的三维坐标
            wx = results.pose_world_landmarks.landmark[i].x
            wy = results.pose_world_landmarks.landmark[i].y
            wz = results.pose_world_landmarks.landmark[i].z
            datatemp[i].append([wx, wy, wz])


# 保存数据的文件
def savecsv(input_path):
    # 获得视频名
    videoname = input_path.split('/')[-1]
    videoname = videoname.split('.')[0]

    print('视频开始处理', input_path)
    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    cap = cv2.VideoCapture(input_path)
    # 创建 保存csv文件的temp
    for i in range(len(pose_name)):
        datatemp[i] = []
        datatemp[i].append(["x", "y", "z"])

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                try:
                    process_frame(frame)
                except:
                    print('error')
                    pass

                if success:
                    # out.write(frame)
                    # 进度条更新一帧
                    pbar.update(1)
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    cap.release()

    # 创建视频数据保存的csv文件夹
    filedir = csvAddress + '/' + videoname
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    # 保存所有骨骼点的数据 33个骨骼点数据
    alldir = filedir + '/allpoints'
    if not os.path.exists(alldir):
        os.makedirs(alldir)
    for i in range(len(pose_name)):
        csvdir = alldir + '/' + pose_name[i] + ".csv"
        with open(csvdir, 'w', newline='') as file:
            writer = csv.writer(file)
            for j in range(len(datatemp[i])):
                writer.writerow(datatemp[i][j])

    # 单独保存上肢骨骼点数据数据 仅两个点位 选取right 14，16号骨骼点
    upperlimbdir = filedir + '/upperlimb'
    if not os.path.exists(upperlimbdir):
        os.makedirs(upperlimbdir)
    for i in range(len(pose_name)):
        if i == 14 or i == 16:
            csvdir = upperlimbdir + '/' + pose_name[i] + ".csv"
            with open(csvdir, 'w', newline='') as file:
                writer = csv.writer(file)
                for j in range(len(datatemp[i])):
                    writer.writerow(datatemp[i][j])

    # 单独保存下肢骨骼点数据数据 仅两个点位 选取right 26，28号骨骼点
    lowerlimbdir = filedir + '/lowerlimb'
    if not os.path.exists(lowerlimbdir):
        os.makedirs(lowerlimbdir)
    for i in range(len(pose_name)):
        if i == 26 or i == 28:
            csvdir = lowerlimbdir + '/' + pose_name[i] + ".csv"
            with open(csvdir, 'w', newline='') as file:
                writer = csv.writer(file)
                for j in range(len(datatemp[i])):
                    writer.writerow(datatemp[i][j])

    print('数据已保存')


# 数据集路径
labAddress = 'F:/everything_lgr/laboratory/project/actionrecognition/datasets'
# domAddress = 'D:/everything'

# 视频文件夹路径
mediaAddress = labAddress + '/media'
# csv文件夹路径
csvAddress = labAddress + '/posecsv'
# 视频文件路径 需要自己指定具体视频
videoAddress = mediaAddress + '/cap.mp4'
# 保存csv
savecsv(input_path=videoAddress)

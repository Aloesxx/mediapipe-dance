import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import os

# 静止，展臂，挥臂，弯臂
upperLimbClass = ["Static", "Unfold", "Wave", "Bend"]
# 静止，踮脚，提膝，曲膝，跨步，跳跃，蹲起
lowerLimbClass = ["Static", "Tiptoe", "Lift", "Bend", "Step", "Jump", "Squat"]
# 33个骨骼点名
poseName = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
            "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
# 暂存骨骼点数据 每三个为一个点位的数据x，y，z
dataTemp = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

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


# 创建保存数据文件夹的函数
def makedir(path):
    for i in range(len(upperLimbClass)):
        for j in range(len(lowerLimbClass)):
            datasetsDir = path + '/' + upperLimbClass[i] + "_" + lowerLimbClass[j]
            if not os.path.exists(datasetsDir):
                os.makedirs(datasetsDir)


# 处理单帧的函数
def process_frame(img):
    # BGR转RGB OpenCV以BGR格式（而不是RGB）读取图像
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型来预测结果
    results = pose.process(img_RGB)
    # 若检测出人体关键点
    # 以一个骨骼点（x,y,z） 三个[]每一组
    for i in range(len(poseName)):
        value = 0
        for n in range(3):
            if results.pose_world_landmarks:
                if n == 0:
                    value = results.pose_world_landmarks.landmark[i].x
                elif n == 1:
                    value = results.pose_world_landmarks.landmark[i].y
                elif n == 2:
                    value = results.pose_world_landmarks.landmark[i].z
            else:
                value = 0
            dataTemp[i * 3 + n].extend([value])


# 保存数据的文件
def savecsv(input_path):
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
    print('视频总帧数为', frame_count - 1)

    cap = cv2.VideoCapture(input_path)
    # 创建和初始化 保存csv文件的temp
    for i in range(len(dataTemp)):
        dataTemp[i] = []

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
                    # 进度条更新一帧
                    pbar.update(1)
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    cap.release()

    rows = zip(dataTemp[0], dataTemp[1], dataTemp[2], dataTemp[3], dataTemp[4], dataTemp[5], dataTemp[6], dataTemp[7], dataTemp[8],
               dataTemp[9], dataTemp[10], dataTemp[11], dataTemp[12], dataTemp[13], dataTemp[14], dataTemp[15], dataTemp[16], dataTemp[17],
               dataTemp[18], dataTemp[19], dataTemp[20], dataTemp[21], dataTemp[22], dataTemp[23], dataTemp[24], dataTemp[25], dataTemp[26],
               dataTemp[27], dataTemp[28], dataTemp[29], dataTemp[30], dataTemp[31], dataTemp[32], dataTemp[33], dataTemp[34], dataTemp[35],
               dataTemp[36], dataTemp[37], dataTemp[38], dataTemp[39], dataTemp[40], dataTemp[41], dataTemp[42], dataTemp[43], dataTemp[44],
               dataTemp[45], dataTemp[46], dataTemp[47], dataTemp[48], dataTemp[49], dataTemp[50], dataTemp[51], dataTemp[52], dataTemp[53],
               dataTemp[54], dataTemp[55], dataTemp[56], dataTemp[57], dataTemp[58], dataTemp[59], dataTemp[60], dataTemp[61], dataTemp[62],
               dataTemp[63], dataTemp[64], dataTemp[65], dataTemp[66], dataTemp[67], dataTemp[68], dataTemp[69], dataTemp[70], dataTemp[71],
               dataTemp[72], dataTemp[73], dataTemp[74], dataTemp[75], dataTemp[76], dataTemp[77], dataTemp[78], dataTemp[79], dataTemp[80],
               dataTemp[81], dataTemp[82], dataTemp[83], dataTemp[84], dataTemp[85], dataTemp[86], dataTemp[87], dataTemp[88], dataTemp[89],
               dataTemp[90], dataTemp[91], dataTemp[92], dataTemp[93], dataTemp[94], dataTemp[95], dataTemp[96], dataTemp[97], dataTemp[98], )

    # 保存在一个文件
    csvdir = input_path.split('.')[0]
    csvdir = csvdir + '.csv'
    with open(csvdir, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

    # 分别保存所有骨骼点的数据 33个骨骼点数据
    # alldir = fileDir + '/allpoints'
    # if not os.path.exists(alldir):
    #     os.makedirs(alldir)
    # for i in range(len(poseName)):
    #     csvdir = alldir + '/' + poseName[i] + ".csv"
    #     with open(csvdir, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         for j in range(len(dataTemp[i])):
    #             writer.writerow(dataTemp[i][j])

    print('数据已保存')


# 数据集路径
csvAddress = 'F:/everything_lgr/laboratory/project/actionrecognition/datasets/posecsv'
domAddress = 'D:/everything/Laboratory/code/posecsv'
# 动作组合名
actionAddress = '/Wave_Jump'
# 视频名
videoName = '/pmla'
# 视频路径
videoAddress = domAddress + actionAddress + videoName + '.mp4'
# 创建数据集文件夹
makedir(domAddress)
# 保存动作片段视频的csv
savecsv(input_path=videoAddress)

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import os

labAddress = 'F:/everything_lgr/media/actrec'

videoAddress = 'D:/everything/VideoandMusicandetc/ActionRecognition'
csvAddress = 'D:/everything/Laboratory/code/posecsv'
pose_name = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
             "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
             "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
             "right_ankle",
             "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
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

"""处理单帧的函数"""


def process_frame(img):
    # 记录开始处理的时间
    # start_time = time.time()
    # 获取图像宽高
    # h, w = img.shape[0], img.shape[1]

    # BGR转RGB OpenCV以BGR格式（而不是RGB）读取图像
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型来预测结果
    results = pose.process(img_RGB)

    if results.pose_world_landmarks:  # 若检测出人体关键点
        for i in range(len(pose_name)):  # 遍历33个关键点
            # 获取关键点的三维坐标
            wx = results.pose_world_landmarks.landmark[i].x
            wy = results.pose_world_landmarks.landmark[i].y
            wz = results.pose_world_landmarks.landmark[i].z
            datatemp[i].append([wx, wy, wz])


"""视频逐帧处理模板   （此函数为模板函数，任何应用只需修改单帧处理函数即可）"""


def save3d(input_path, file_type='mp4'):
    videoname = input_path.split('/')[-1]
    videoname = videoname.split('.')[0]  # 格式为 /xxx
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

    # cv2.namedWindow('your name')
    cap = cv2.VideoCapture(input_path)
    # 获取视频窗口宽、高
    # frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    """
    # fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数，注意：字符顺序不能弄混
    # cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
    # cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    """
    # if file_type == 'mp4':
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 子豪兄是mp4的
    # if file_type == 'flv':
    #     fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

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

    filedir = csvAddress + '/' + videoname
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    for i in range(len(pose_name)):
        csvdir = filedir + '/' + pose_name[i] + ".csv"
        with open(csvdir, 'w', newline='') as file:
            writer = csv.writer(file)
            for j in range(len(datatemp[i])):
                writer.writerow(datatemp[i][j])

    cv2.destroyAllWindows()
    # out.release()
    cap.release()
    print('数据已保存')


save3d(input_path=videoAddress + '/cap.mp4', file_type='mp4')

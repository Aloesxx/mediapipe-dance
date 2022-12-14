# import cv2
# import mediapipe as mp
# from vpython import *
#
#
# # mediapipe 模型变量初始化（你教我的，要整理变量）
# def mediapipe_varibles_init():
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
#     mp_drawing = mp.solutions.drawing_utils
#     return pose, mp_pose, mp_drawing
#
#
# # vpython（三维画图）模型变量初始化
# def vpython_variables_init():
#     points = []
#     boxs = []
#     ids = [[12, 14, 16], [11, 13, 15], [12, 24, 26, 28, 30, 32, 28],
#            [11, 23, 25, 27, 29, 31, 27], [12, 11], [24, 23]]
#     c = []
#
#     for x in range(33):
#         points.append(sphere(radius=5, pos=vector(0, -50, 0)))
#         c.append(curve(retain=2, radius=4))
#     return points, boxs, ids, c, hpointsr, hpointsl
#
#
# # 在3D里画出骨架的函数
# def draw_3d_pose():
#     results = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
#     if results.pose_world_landmarks:
#         for i in range(11, 33):
#             # 手掌的3个点
#             if i != 18 and i != 20 and i != 22 and i != 17 and i != 19 and i != 21:
#                 # 通过cap.get（3）和cap.get（4）来检查帧的宽度和高度，默认的值是640x480。
#                 points[i].pos.x = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * -cap.get(3)
#                 points[i].pos.y = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * -cap.get(4)
#                 points[i].pos.z = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * -cap.get(3)
#         for n in range(2):
#             for i in range(2):
#                 c[i + 2 * n].append(vector(points[ids[n][i]].pos.x, points[ids[n][i]].pos.y, points[ids[n][i]].pos.z),
#                                     vector(points[ids[n][i + 1]].pos.x, points[ids[n][i + 1]].pos.y,
#                                            points[ids[n][i + 1]].pos.z), retaine=2)
#         for n in range(2, 4):
#             for i in range(6):
#                 c[i + 6 * n].append(vector(points[ids[n][i]].pos.x, points[ids[n][i]].pos.y, points[ids[n][i]].pos.z),
#                                     vector(points[ids[n][i + 1]].pos.x, points[ids[n][i + 1]].pos.y,
#                                            points[ids[n][i + 1]].pos.z), retaine=2)
#         for n in range(4, 6):
#             for i in range(1):
#                 c[i + 2 * n].append(vector(points[ids[n][i]].pos.x, points[ids[n][i]].pos.y, points[ids[n][i]].pos.z),
#                                     vector(points[ids[n][i + 1]].pos.x, points[ids[n][i + 1]].pos.y,
#                                            points[ids[n][i + 1]].pos.z), retaine=2)
#     mp_drawing.draw_landmarks(image=f, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
#
#
# # 窗口关闭函数
# def clos_def():
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # 获取变量
# points, boxs, ids, c, hpointsr, hpointsl = vpython_variables_init()
# pose, mp_pose, mp_drawing = mediapipe_varibles_init()
# # 打开摄像头，0是第一个摄像头，如果想换一个摄像头请改变这个数字
# cap = cv2.VideoCapture(0)
# while True:
#     # 获取每一帧的图像
#     _, f = cap.read()
#     # vpython里的一个函数，用来调整3D中的FPS
#     rate(150)
#     # 调用在3D里画出骨架的函数
#     draw_3d_pose()
#     # 在每一帧里画骨架
#     # 显示每一帧
#     cv2.imshow('real_time', f)
#     # 检测是否要关闭窗口
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # 调用窗口关闭函数
# clos_def()
# a = ["A", "B", "C"]

# x = [[],[],[]]
# print(len(x))
#
# for i in range(3):
#     for j in range(3):
#         x[i].extend([j+1])
# print(x)

import csv

filePath = 'temp.csv'
list1 = [1, 2, 3, 4]
list2 = [4, 2, 3, 4]
list3 = [5, 6, 7, 4]
list4 = [8, 2, 9, 6]
rows = zip(list1, list2, list3, list4)
with open(filePath, "w", newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

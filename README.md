# mediapipe-dance
# 提取视频骨骼点
## mediapipe
使用的是mediapipe——zihaoDance
### 注意事项
1. 我自己笔记本是py3.8，实验室电脑是py3.9，所以手动下载安装mediapipe切勿装错版本。
2. 笔记本电脑Pycharm中使用cv2代码无法自动补全，解决办法参考链接https://blog.csdn.net/Z2572862506/article/details/125243384
3. 导入模型,第一次运行mediapipe代码的时候有时候会下载模型，但是有时候因为网络问题，可能下载不下来，报错：远程链接拒绝。这是因为现在是从谷歌网盘下，国内你懂的。参考博客解决办法 https://blog.csdn.net/m0_57110410/article/details/125538796
## 构建分类器
### 训练集数据结构
俩个文件夹，分别存放深蹲的两个极端情况--站着和完全蹲下。
而且都是周围360°拍摄的照片，完整的人，不存在遮挡
* squat_dataset
    * up
        * image_001.jpg 
        * image_002.jpg 
        ...
    * down
        * image_001.jpg 
        * image_002.jpg 
        ...
## 其他说明
可视化模块的字体部分，由于源代码要连外网下，所以灭活了一部分，如有需要可后续自行修改
Basic_A-F下需要建两个文件夹，分别为“videos”，“videos——processed”
G-squat_detect下需要建一个文件夹，“squat_videos”
## package
* python               >= 3.8
* opencv-python        4.5.1.48
* mediapipe            0.8.11
* numpy                1.22.1

# 数据集说明
## 动作分类标签说明（详见实验设计方案.pdf）
### 手臂
静止，展臂，挥臂，弯臂。
### 腿部
静止，踮脚，提膝，曲膝，跨步，跳跃，蹲起。
## 数据存放说明
### 总文件夹
以标签（标签有4*7=28个，手和脚的动作组合为一个标签）作为文件夹名。
该文件夹中需存放：不同的视频文件夹。因为有的视频该动作幅度大，不太容易识别出，如果模型效果不好就不要拿去训练
### 数据文件夹/标签文件夹
数据/标签文件夹中需存放：
1. 对应标签的动作片段录屏。从开始到结束的一次动作。
2. 动作片段录屏的数据。33个骨骼点全存，需要哪个用哪个。一个点有3轴数据，所以每3列存放一个点的数据（3列中的每一行就是该点在动作片段的每一帧的数据）。
## 文件内容举例说明
例如该项目路径下的datasets文件夹，里面存放了Wave_Jump文件夹（文件夹名是标签），再里面存放了pmla.mp4和pmla.csv两个文件对应于Wave_Jump的动作片段录屏和提取出来的数据文件。
csv文件有99列（33个骨骼点×每个点3个轴xyz的数据），行数为该动作片段录屏的帧数。

## 数据集采集流程
### 1. 获取动作片段视频，将该视频片段保存在对应的标签文件夹中
视频就用咱们找的，见datasets/选取的视频集.xlsx。
可以直接选择录屏（如QQ录屏）或者利用剪辑工具。视频内容为一个（注意是一个，不是几十秒的重复在做同一动作）完整的动作就行。比如开合跳动作，开始是手脚并拢，到手举过头顶和脚跨开，结束是再次手脚并拢，这算一个动作。
### 2. 利用项目中的savecsv.py文件进行提取骨骼点数据并保存
需要自行修改里面的路径参数

# 动作识别模型
考虑用什么模型吧
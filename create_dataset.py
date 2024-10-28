import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# 数据目录
DATA_DIR = './data'

# 初始化数据和标签列表
data = []
labels = []

# 速度和加速度的计算函数
def calculate_velocity_and_acceleration(previous_points, current_points):
    velocity = []
    acceleration = []
    if previous_points is not None:
        # 计算速度
        velocity = [np.linalg.norm(np.array(current) - np.array(previous)) for current, previous in zip(current_points, previous_points)]
    if len(velocity) > 0:
        # 计算加速度
        acceleration = [np.linalg.norm(np.array(velocity[i]) - np.array(velocity[i - 1])) for i in range(1, len(velocity))]
        # 为了对齐长度，在加速度数组前面加一个0（因为首帧没有加速度）
        acceleration.insert(0, 0)
    return velocity, acceleration

# 遍历数据文件夹
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        
        # 记录 x、y、z 位置以计算速度和加速度
        previous_points = None
        current_points = []

        # 读取图像
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取手部关键点
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 遍历每个关键点
                for landmark in hand_landmarks.landmark:
                    current_points.append((landmark.x, landmark.y, landmark.z))
                
                # 计算速度和加速度
                velocity, acceleration = calculate_velocity_and_acceleration(previous_points, current_points)
                
                # 位置特征
                for point in current_points:
                    data_aux.extend(point)
                
                # 速度特征
                data_aux.extend(velocity if velocity else [0] * len(current_points))  # 如果没有前一帧，则速度为0
                
                # 加速度特征
                data_aux.extend(acceleration if acceleration else [0] * len(current_points))  # 如果没有速度，则加速度为0
                
                # 更新前一帧的点
                previous_points = current_points
        
            data.append(data_aux)
            labels.append(dir_)

# 保存数据到 pickle 文件
with open('data_with_velocity_acceleration.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

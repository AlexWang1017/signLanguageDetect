import os
import numpy as np
import pickle

# 配置参数
DATA_DIR = './data'
OUTPUT_FILE = 'data_with_velocity_acceleration.pickle'
GESTURE_LABELS = ['5_Static', 'Hello_Dynamic']
TIMESTEPS = 30  # 每个样本的时间步

data = []
labels = []

for label in GESTURE_LABELS:
    gesture_dir = os.path.join(DATA_DIR, label)
    frames = sorted([os.path.join(gesture_dir, f) for f in os.listdir(gesture_dir) if f.endswith('.npy')])

    for i in range(len(frames) - TIMESTEPS):
        sample_data = []
        for j in range(i, i + TIMESTEPS):
            keypoints = np.load(frames[j])

            # 计算速度和加速度
            if j > i:
                prev_keypoints = np.load(frames[j - 1])
                velocity = keypoints - prev_keypoints
                if j > i + 1:
                    prev_velocity = sample_data[-1][-3:]  # 上一个速度
                    acceleration = velocity - prev_velocity
                else:
                    acceleration = np.zeros_like(velocity)
            else:
                velocity = np.zeros_like(keypoints)
                acceleration = np.zeros_like(keypoints)

            # 合并位置、速度和加速度特征
            features = np.concatenate([keypoints, velocity, acceleration])
            sample_data.append(features)

        data.append(sample_data)
        labels.append(label)

# 转换为数组并保存
data = np.array(data)
labels = np.array(labels)
data_dict = {'data': data, 'labels': labels}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data_dict, f)

print("数据集已创建并保存。")

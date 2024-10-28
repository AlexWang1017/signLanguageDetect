import os
import numpy as np
import pickle

# 配置参数
DATA_DIR = './data'
OUTPUT_FILE = 'data_with_velocity_acceleration.pickle'
GESTURE_LABELS = ['5_Static', 'Hello_Dynamic']
label_mapping = {label: idx for idx, label in enumerate(GESTURE_LABELS)}  # 标签映射
TIMESTEPS = 30  # 每个样本的时间步

data = []
labels = []

for label in GESTURE_LABELS:
    gesture_dir = os.path.join(DATA_DIR, label)
    frames = sorted([os.path.join(gesture_dir, f) for f in os.listdir(gesture_dir) if f.endswith('.npy')])

    # 预加载所有关键点数据
    all_keypoints = [np.load(frame) for frame in frames]

    # 生成样本
    for i in range(len(all_keypoints) - TIMESTEPS):
        sample_data = []

        for j in range(TIMESTEPS):
            keypoints = all_keypoints[i + j]

            # 默认速度和加速度为零向量
            velocity = np.zeros_like(keypoints)
            acceleration = np.zeros_like(keypoints)

            # 从第二帧开始计算速度和加速度
            if j > 0:
                prev_keypoints = all_keypoints[i + j - 1]
                velocity = keypoints - prev_keypoints

                # 确保有前一帧的速度信息
                if j > 1:
                    prev_velocity = sample_data[-1][len(keypoints):2 * len(keypoints)]
                    acceleration = velocity - prev_velocity

            # 合并位置、速度和加速度特征
            features = np.concatenate([keypoints, velocity, acceleration])
            sample_data.append(features)

        data.append(sample_data)
        labels.append(label_mapping[label])

# 转换为数组并保存
data = np.array(data, dtype=object)
labels = np.array(labels, dtype=np.int32)
data_dict = {'data': data, 'labels': labels}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data_dict, f)

print("数据集已创建并保存。")

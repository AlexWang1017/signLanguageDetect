import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 配置参数
DATA_DIR = './data'
GESTURE_LABELS = {0: '5_Static', 1: 'Hello_Dynamic'}  # 手势标签
FRAMES_PER_GESTURE = 60  # 每个手势采集的帧数，增加帧数以捕获动态手势
FRAME_INTERVAL = 50  # 每帧间隔时间，单位为毫秒
gesture_id = 0  # 手势 ID，用于标记不同手势

# 创建 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

# 初始化摄像头
cap = cv2.VideoCapture(0)

print("按下空格键开始采集手势...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 等待用户按下空格键以开始采集手势数据
    if cv2.waitKey(1) & 0xFF == ord(' '):
        gesture_name = GESTURE_LABELS[gesture_id]
        gesture_dir = os.path.join(DATA_DIR, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        print(f"开始采集手势: {gesture_name}")

        # 采集指定帧数
        for frame_id in range(FRAMES_PER_GESTURE):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            data = []  # 存储关键点数据
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        data.extend([landmark.x, landmark.y, landmark.z])

            # 保存帧数据
            data = np.array(data)
            np.save(os.path.join(gesture_dir, f'{gesture_name}_{frame_id}.npy'), data)

            cv2.putText(frame, f'Collecting {gesture_name}: Frame {frame_id + 1}/{FRAMES_PER_GESTURE}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('frame', frame)

            cv2.waitKey(FRAME_INTERVAL)  # 每帧间隔

        print(f"手势 {gesture_name} 采集完成！")
        
        # 切换到下一个手势，循环标签
        gesture_id = (gesture_id + 1) % len(GESTURE_LABELS)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

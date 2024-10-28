import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# 加载模型
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8)

# 定义标签字典
labels_dict = {0: '5 Static', 1: 'Hello Dynamic'}

# 配置参数
static_threshold = 0.05  # 静态手势的变化阈值
gesture_buffer_time = 0.5  # 动态手势缓冲时间，单位秒
window_size = 5  # 滑动窗口的大小，包含5帧
dynamic_confirmation_frames = 3  # 动态手势确认的连续帧数

# 初始化滑动窗口和缓冲队列
window_data = deque(maxlen=window_size)  # 用于滑动窗口的队列
previous_data = None  # 保存上一帧数据
buffer_start_time = None  # 缓冲区开始时间
dynamic_frame_count = 0  # 连续检测到动态手势的帧数

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []  # 新增 z 轴数据的容器

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理当前帧
    results = hands.process(frame_rgb)
    gesture_type = "Unknown Gesture"
    buffered_gesture = "5 Static"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                z_.append(landmark.z)

            if x_ and y_ and z_:
                for i in range(len(x_)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                    data_aux.append(z_[i])

            # 添加到滑动窗口
            window_data.append(data_aux)

            # 判断是静态还是动态手势
            if previous_data is not None:
                # 计算位置变化
                movement = np.linalg.norm(np.array(data_aux) - np.array(previous_data))
                if movement < static_threshold:
                    # 静态手势
                    gesture_type = "Static Gesture"
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_label = int(prediction[0])
                    if predicted_label == 0:
                        buffered_gesture = "5 Static"
                    else:
                        buffered_gesture = "Unknown Static"
                    dynamic_frame_count = 0  # 重置动态计数
                else:
                    # 动态手势检测逻辑
                    gesture_type = "Dynamic Gesture"
                    if buffer_start_time is None:
                        buffer_start_time = time.time()
                    
                    # 检查缓冲时间
                    if time.time() - buffer_start_time >= gesture_buffer_time:
                        dynamic_frame_count += 1  # 增加动态确认帧数
                        if dynamic_frame_count >= dynamic_confirmation_frames:
                            buffered_gesture = "Hello Dynamic"
                        else:
                            buffered_gesture = "Unknown Dynamic"
                    else:
                        buffered_gesture = "Dynamic (Buffering)"

            # 更新 previous_data
            previous_data = data_aux

    # 显示结果在右上角
    text = f"{gesture_type}: {buffered_gesture}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    text_x = W - text_size[0] - 10
    text_y = 30

    # 绘制背景矩形
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    # 显示窗口
    cv2.imshow('frame', frame)

    # 检查是否按下空格键退出
    key = cv2.waitKey(10)
    if key == 32:
        break

cap.release()
cv2.destroyAllWindows()

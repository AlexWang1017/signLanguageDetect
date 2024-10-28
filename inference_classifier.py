import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from keras.models import load_model

# 加载模型
model = load_model('lstm_gesture_model.h5')

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8)

# 定义标签字典
labels_dict = {0: '5 Static', 1: 'Hello Dynamic'}

# 配置参数
static_threshold = 0.02  # 降低静态手势的变化阈值
dynamic_threshold = 0.03  # 降低动态手势检测的移动量阈值
window_size = 30  # 滑动窗口的大小，包含30帧
features = 3      # 每帧包含 x, y, z 三个特征
movement=0

# 初始化滑动窗口
window_data = deque(maxlen=window_size)  # 滑动窗口用于存储帧数据
previous_data = None  # 保存上一帧数据

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

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

            # 获取手部关键点数据
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                z_.append(landmark.z)

            # 归一化关键点并添加到 data_aux
            if x_ and y_ and z_:
                for i in range(len(x_)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                    data_aux.append(z_[i])

            # 添加到滑动窗口
            if len(data_aux) == window_size * features:
                window_data.append(data_aux)

                # 确保 previous_data 已初始化
                if previous_data is None:
                    previous_data = np.zeros_like(data_aux)

                # 检查滑动窗口大小，确保满足模型输入要求
                if len(window_data) == window_size:
                    # 将滑动窗口中的数据转为 numpy 数组
                    input_data = np.array(window_data).reshape(1, window_size, features)

                    # 动态检测逻辑：计算位置变化
                    movement = np.linalg.norm(np.array(data_aux) - np.array(previous_data))
                    print(f"Movement detected: {movement}", flush=True)  # 调试输出，查看移动量

                    if movement >= dynamic_threshold:
                        gesture_type = "Dynamic Gesture"
                        # 进行预测
                        prediction = model.predict(input_data)
                        predicted_label = np.argmax(prediction)
                        buffered_gesture = labels_dict.get(predicted_label, "Hello Dynamic")
                    else:
                        gesture_type = "Static Gesture"
                        buffered_gesture = "5 Static"
                    previous_data = data_aux  # 更新上一帧的数据

            # 重置 data_aux
            data_aux = []

    # 显示结果在右上角
    text = f"{gesture_type}: {buffered_gesture}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    text_x = W - text_size[0] - 10
    text_y = 30
    movement_text = f"Movement: {movement:.4f}"

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
 
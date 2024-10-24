import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# 加載模型
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# 定義標籤字典
labels_dict = {0: '5 Static', 1: 'Hello'}  # 將靜態手勢和動態手勢的標籤定義清楚
static_threshold = 0.05  # 設定靜態閾值
previous_data = None  # 儲存上一幀的數據
detecting_gesture = False  # 控制是否正在檢測手勢
gesture_buffer_time = 2  # 緩衝時間（秒）
gesture_start_time = None  # 手勢開始時間
buffered_gesture = "5"  # 初始顯示為 5

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []  # 新增 z 軸數據的容器

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 初始化手勢類型
    gesture_type = "Unknown Gesture"

    # 檢查是否開始偵測手勢
    if detecting_gesture:
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # 獲取每個手部地標的 x、y 和 z 值
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z  # 獲取 z 軸數據

                    x_.append(x)
                    y_.append(y)
                    z_.append(z)

                # 歸一化 x 和 y 的數據
                if x_:
                    for x_val in x_:
                        data_aux.append(x_val - min(x_))  # x 軸歸一化
                if y_:
                    for y_val in y_:
                        data_aux.append(y_val - min(y_))  # y 軸歸一化
                if z_:  # 如果有 z 軸數據
                    for z_val in z_:
                        data_aux.append(z_val)  # 直接添加 z 軸數據，根據需求進行處理

            # 繪製邊界框
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # 判斷是靜態還是動態手勢
            if previous_data is not None:
                # 計算位置變化
                movement = np.linalg.norm(np.array(data_aux) - np.array(previous_data))
                if movement < static_threshold:
                    # 如果是靜態手勢
                    gesture_type = "Static Gesture"
                    # 進行靜態手勢的預測
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_label = int(prediction[0])
                    if predicted_label == 0:  # 如果預測是靜態手勢 5
                        buffered_gesture = "5"  # 強制顯示 5
                    else:
                        buffered_gesture = "5"  # 只顯示數字 5
                else:
                    # 如果是動態手勢
                    gesture_type = "Dynamic Gesture"
                    buffered_gesture = "Hello"  # 動態手勢顯示為 Hello

            # 確保上一幀的數據被更新
            previous_data = data_aux

    # 繪製結果並固定在右上角
    text = f"{gesture_type}: {buffered_gesture}"  # 使用當前手勢
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]

    # 計算文本顯示位置
    text_x = W - text_size[0] - 10  # 右側邊距
    text_y = 30  # Y座標（距離上邊距）

    # 繪製背景矩形
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                   (text_x + text_size[0] + 10, text_y + 10), 
                   (0, 0, 0), -1)
    # 繪製文本
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    # 顯示視窗
    cv2.imshow('frame', frame)

    # 檢查是否按下 Q 鍵開始偵測手勢或空白鍵關閉視窗
    key = cv2.waitKey(10)
    if key == ord('q'):
        detecting_gesture = not detecting_gesture  # 切換手勢偵測狀態
    elif key == 32:  # 32 是空白鍵的 ASCII 值
        break  # 跳出循環，關閉視窗

cap.release()
cv2.destroyAllWindows()

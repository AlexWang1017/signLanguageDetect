import os
import cv2
import time

# 设置数据保存目录
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 设置采集参数
gesture_labels = {0: '5', 1: 'HELLO'}  # 定义手势标签
frames_per_gesture = 100  # 每个手势采集的帧数，增加到100帧
gesture_delay = 2  # 每个手势采集之间的延迟（秒）
frame_interval = 50  # 每帧间隔时间，单位为毫秒

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 提示用户按下空白键以开始采集
print("请准备好手势，按下空白键以开始采集...")

# 等待按下空白键
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 显示提示文本
    cv2.putText(frame, 'Press SPACE to start capturing', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    # 检查空白键（ASCII码 32）
    if cv2.waitKey(10) == 32:
        print("开始采集数据...")
        break

# 开始采集数据
for label, gesture_name in gesture_labels.items():
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    
    print(f'准备采集手势: {gesture_name}')
    time.sleep(gesture_delay)  # 采集前延迟准备

    frame_counter = 0  # 帧计数器
    while frame_counter < frames_per_gesture:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示当前采集状态
        cv2.putText(frame, f'Collecting {gesture_name}: Frame {frame_counter + 1}/{frames_per_gesture}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        # 保存帧
        frame_path = os.path.join(gesture_dir, f'{gesture_name}_{frame_counter}.jpg')
        cv2.imwrite(frame_path, frame)
        
        frame_counter += 1
        cv2.waitKey(frame_interval)  # 每帧间隔50ms，增加采样时间

    print(f'{gesture_name} 手势采集完成，采集了 {frames_per_gesture} 帧')
    time.sleep(gesture_delay)  # 切换手势前延迟

cap.release()
cv2.destroyAllWindows()

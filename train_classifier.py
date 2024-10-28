import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# 加载数据
with open('data_with_velocity_acceleration.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# 数据预处理
data = np.asarray(data_dict['data'], dtype=np.float32)  # 确保为 float32 类型
labels = np.asarray(data_dict['labels'])

# 对标签进行编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded).astype(np.float32)  # 转换为 float32 类型

# 检查数据形状
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels_one_hot.shape}")

# 数据集拆分
x_train, x_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, shuffle=True, stratify=labels_encoded)

# 创建 LSTM 模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 保存模型
model.save('lstm_gesture_model.h5')

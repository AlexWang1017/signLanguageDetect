import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 加载数据
with open('data_with_velocity_acceleration.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# 数据预处理
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 对标签进行编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# 将数据重塑为 [samples, timesteps, features] 以适应 LSTM
timesteps = 30  # 假设每个样本包含30帧
features = data.shape[1] // timesteps
data_reshaped = data.reshape(-1, timesteps, features)

# 数据集拆分
x_train, x_test, y_train, y_test = train_test_split(data_reshaped, labels_one_hot, test_size=0.2, shuffle=True, stratify=labels_encoded)

# 创建 LSTM 模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(timesteps, features)))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 保存模型
model.save('lstm_gesture_model.h5')

from keras.utils import to_categorical
import numpy as np

# 定义一个标签数组
labels = np.array([0, 1, 2, 3])

# 使用 to_categorical 将标签转换为 one-hot 编码
one_hot_labels = to_categorical(labels)

# 输出结果
print("Original labels:", labels)
print("One-hot encoded labels:\n", one_hot_labels)
from sklearn import svm
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
train_num = 60000
test_num = 1000

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 获取一个支持向量机模型
predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
# 把数据丢进去
predictor.fit(x_train[:train_num], y_train[:train_num])
# 预测结果
result = predictor.predict(x_test[:test_num])
# 准确率估计
accurancy = np.sum(np.equal(result, y_test[:test_num])) / test_num
print(accurancy)

from libsvm.svm import *
from libsvm.svmutil import *

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
def img2vector(train,label,num):
    """
    将28*28的二进制图像转换为1x784向量。
    Parameters:
        train - 数据集
        label - 对应数据集的标签
        num - 数据集的大小
    Returns:
        returnVect - 返回的二进制图像的1x784向量
    """
    train_labels = []
    trainingMat = np.zeros((num, 784))
    for i in range(num):
        trainingMat[i, :] = train[i].reshape(1, 784)
        labelM = label[i]
        for j in range(len(labelM)):
            if labelM[j] == 1:
                if j == 9:
                    train_labels.append(1)
                else:
                    train_labels.append(-1)
                break
    return trainingMat,train_labels

def loadImages(train_num,test_num):
    mnist = input_data.read_data_sets('../datasets/mnist', one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    X_train = X_train[0:train_num]
    Y_train = Y_train[0:train_num]
    X_test = X_test[0:test_num]
    Y_test = Y_test[0:test_num]
    train_Mat, train_labels = img2vector(X_train,Y_train,train_num)
    test_Mat, test_labels = img2vector(X_test,Y_test,test_num)

    return train_Mat, train_labels,test_Mat, test_labels

if __name__ == '__main__':
    time_start = time.time()
    x_train,y_train ,x_test, y_test = loadImages(5000,1000)
    model = svm_train(y_train, x_train )
    print('test:')
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
    time_end = time.time()
    m, s = divmod(time_end - time_start, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))
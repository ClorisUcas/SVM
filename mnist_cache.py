# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt

import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
import time


class optStruct:
    """
    数据结构，维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup,K_num):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 数据标签
        self.C = C  # 松弛变量
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m,K_num)))  # 初始化核K
        self.K_cache = {}
        self.kTup=kTup

def kernelTrans(X, A, kTup):
    """
    通过核函数将数据转换更高维的空间
    Parameters：
        X - 数据矩阵
        A - 单个数据的向量
        kTup - 包含核函数信息的元组
    Returns:
        K - 计算的核K
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积。
    elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # 计算高斯核K
    else:
        raise NameError('核函数无法识别')
    return K  # 返回计算的核K

def calcEk(oS, k):
    """
    计算误差
    Parameters：
        oS - 数据结构
        k - 标号为k的数据
    Returns:
        Ek - 标号为k的数据误差
    """
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    """
    函数说明:随机选择alpha_j的索引值

    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
    """
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def selectJ(i, oS, Ei,count):
    """
    内循环启发方式2
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
        Ei - 标号为i的数据误差
        count - 随机选择时在缓存的数据量中进行选择
    Returns:
        j, maxK - 标号为j或maxK的数据的索引值
        Ej - 标号为j的数据误差
        temp - 标号为j的oS.K
    """
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0  # 初始化
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    m, n =np.shape(oS.K)
    temp = 0
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i: continue  # 不计算i,浪费时间
            if k> n:
                temp = kernelTrans(oS.X, oS.X[k, :], oS.kTup)
                fXk = float(np.multiply(oS.alphas, oS.labelMat).T * temp + oS.b)
                Ek = fXk - float(oS.labelMat[k])
            else:
                Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej, temp  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, count)  # 随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  # 计算Ej
    return j, Ej, temp  # j,Ej


def updateEk(oS, k):
    """
    计算Ek,并更新误差缓存
    Parameters：
        oS - 数据结构
        k - 标号为k的数据的索引值
    Returns:
        无
    """
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存

def clipAlpha(aj, H, L):
    """
    修剪alpha_j
    Parameters:
        aj - alpha_j的值
        H - alpha上限
        L - alpha下限
    Returns:
        aj - 修剪后的alpah_j的值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerL(i, oS,count=-1,K_cache_i = False):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
        K_cache_i - 判断是否需要缓存非边界的核函数
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        if count== -1:
            count = oS.m
        j, Ej, temp = selectJ(i, oS, Ei,count)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        if temp != 0: oS.K[j,j] = temp
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("alpha_j变化太小")
            return 0
        # 更新Ej至误差缓存
        updateEk(oS, j)
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)

        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
            if K_cache_i == True:  oS.K_cache[i] = oS.K
        if (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
            if K_cache_i == True:  oS.K_cache[j] = oS.K
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0),K_num=1000):
    """
    完整的线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
        kTup - 包含核函数信息的元组
        K_num - 存储核函数列表的长度
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup,K_num)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True;
    alphaPairsChanged = 0
    # iteration = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if  entireSet:
            temp = int(np.ceil(oS.m / K_num))
            for num in range(temp):
                if temp == 1:
                    count = oS.m
                elif num != temp:
                    count = K_num
                else:
                    count = oS.m - K_num * (temp - 1)
                for i in range(count):
                    # for j in range(oS.m):
                    oS.K[:, i] = kernelTrans(oS.X, oS.X[i, :], kTup)
                for i in range(count):
                    alphaPairsChanged += innerL(i, oS,count,K_cache_i=True)  # 使用优化的SMO算法
            iter +=1

        else:  # 遍历非边界值
            # nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            # for i in nonBoundIs:
            for i in oS.K_cache.keys():
                oS.K = oS.K_cache[i]
                alphaPairsChanged += innerL(i, oS)
                # print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
            oS.K_cache={}
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        # print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas


def img2vector(train, label, num):
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
    return trainingMat, train_labels

def loadImages(train_num, test_num):
    mnist = input_data.read_data_sets('../datasets/mnist', one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    X_train = X_train[0:train_num]
    Y_train = Y_train[0:train_num]
    X_test = X_test[0:test_num]
    Y_test = Y_test[0:test_num]
    train_Mat, train_labels = img2vector(X_train, Y_train, train_num)
    test_Mat, test_labels = img2vector(X_test, Y_test, test_num)

    return train_Mat, train_labels, test_Mat, test_labels

def testDigits(kTup=('rbf', 10)):
    """
    测试函数
    Parameters:
        kTup - 包含核函数信息的元组
    Returns:
        无
    """
    time_start = time.time()
    dataArr, labelArr, test_dataArr, test_labelArr = loadImages(10000, 1000)
    b, alphas = smoP(dataArr, labelArr, 100, 0.00001, 10, kTup,K_num=100)
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    # print("支持向量个数:%d" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount) / m))

    errorCount = 0
    datMat = np.mat(test_dataArr)
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(test_labelArr[i]): errorCount += 1
    print("测试集错误率: %.2f%%" % (float(errorCount) / m))
    time_end = time.time()
    m, s = divmod(time_end - time_start, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))


if __name__ == '__main__':
    testDigits()
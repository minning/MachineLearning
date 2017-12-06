# coding:utf-8
'''
    Author:minning
    Date:2017/12/6
    代码目的：逻辑回归的代码实现

'''

import numpy as np
import os


def loadData(folderPath):
    '''
        读取folderPath文件夹下的文件，并将其处理成矩阵形式输出
        dataArray，lableArray分别对应于数据的X，y
    :param folderPath:
    :return:
    '''
    filesList = os.listdir(folderPath)
    m = len(filesList)
    dataArray = np.zeros((m, 1024))
    lableArray = np.zeros((m, 1))

    for fileIndex, fileName in enumerate(filesList):
        fileAbsPath = os.path.join(folderPath, fileName)
        fileArray = np.zeros((1, 1024))
        with open(fileAbsPath) as file:
            for lineIndex, line in enumerate(file):
                for itemIndex, item in enumerate(line.strip()):
                    fileArray[0, lineIndex * 32 + itemIndex] = int(item)
        dataArray[fileIndex, :] = fileArray

        fileLable = int(fileName.split('.')[0].split('_')[0])
        lableArray[fileIndex] = fileLable

    return dataArray, lableArray


def sigmoid(XMat):
    '''
    Sigmoid函数的代码实现，需要注意的是，numpy中可以对一整个矩阵求函数变换，
           这是numpy的特性，和广播类似，可以加快运算速度
    :param XMat: 待运算的数，可以是int，也可以是matrix、ndarray
    :return:
    '''
    return 1.0 / (1 + np.exp(-1.0 * XMat))


def graDsent(dataArray, lableArray, learning_rate=0.05, maxCycle=20):
    '''
        梯度下降法求解参数训练模型
    :param dataArray: 对标模型的 X
    :param lableArray:  对标模型的 y
    :param learning_rate: 学习率
    :param maxCycle: 最大迭代轮数
    :return: 训练好的模型参数
    '''
    dataMat = np.mat(dataArray)  # (301, 1024)
    lableMat = np.mat(lableArray)  # (301, 1)
    para = np.ones((dataMat.shape[1], 1))  # (1024, 1)

    for i in range(maxCycle):
        f_x = sigmoid(dataMat * para)
        error = lableMat - f_x
        para = para + learning_rate * dataMat.transpose() * error

    return para


def classify(dataArray, lableArray, para):
    '''
    根据训练好的模型进行测试，输出测试结果
    :param dataArray: 数据的特征，对标测试数据的 X
    :param lableArray:  数据的标签， 对标测试数据的 y
    :param para: 训练好的模型参数
    :return: 错误率
    '''
    dataMat = np.mat(dataArray)  # (80, 1024)
    lableMat = np.mat(lableArray)  # (80, 1)
    preMat = dataMat * para
    preLable = sigmoid(preMat)

    i = 0
    for index, preOneLable in enumerate(preLable):
        preOneLableItem = preOneLable[0]
        if preOneLableItem > 0.5:
            print "{} is predict to 1".format(lableMat[index, 0])
            if int(lableMat[index, 0]) != 1:
                i += 1
        else:
            print "{} is predict to 0".format(lableMat[index, 0])
            if int(lableMat[index, 0]) != 0:
                i += 1

    return 1.0 * i / lableMat.shape[0]


def main():
    dataArray, lableArray = loadData('train')
    testData, testLable = loadData('test')
    para = graDsent(dataArray, lableArray)
    result = classify(testData, testLable, para)
    print "The error rate is {}".format(result)


if __name__ == "__main__":
    main()

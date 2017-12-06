# coding:utf-8
'''
    Author:minning
    Date:2017/12/6
    代码目的：
    
'''

import numpy as np
import os


def loadData(folderPath):
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
    return 1.0 / (1 + np.exp(-1.0 * XMat))


def graDsent(dataArray, lableArray, learning_rate=0.05, maxCycle=20):
    dataMat = np.mat(dataArray)  # (301, 1024)
    lableMat = np.mat(lableArray)  # (301, 1)
    para = np.ones((dataMat.shape[1], 1))  # (1024, 1)

    for i in range(maxCycle):
        f_x = sigmoid(dataMat * para)
        error = lableMat - f_x
        para = para + learning_rate * dataMat.transpose() * error

    return para


def classify(dataArray, lableArray, para):
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
    # print "dataArray shape : {}".format(dataArray.shape)
    # print "lableArray shape : {}".format(lableArray.shape)
    # print "testData shape : {}".format(testData.shape)
    # print "testLable shape : {}".format(testLable.shape)
    para = graDsent(dataArray, lableArray)

    result = classify(testData, testLable, para)
    print "The error rate is {}".format(result)


if __name__ == "__main__":
    main()

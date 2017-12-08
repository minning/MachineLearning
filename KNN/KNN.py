# coding:utf-8
'''
    Author:minning
    Date:2017/12/8
    代码目的：KNN的代码实现，完成数字识别的分类任务
    
'''
import numpy as np
import os
from collections import Counter


def img2Vec(imgPath):
    '''
    将输入图像转化为矩阵输出
    :param imgPath: 输入图像的地址
    :return: 输入图像对应的矩阵输出
    '''
    vec = np.zeros((1, 1024))
    with open(imgPath, 'r') as img:
        for lineIndex, line in enumerate(img):
            for itemIndex, item in enumerate(line.strip()):
                vec[0, lineIndex * 32 + itemIndex] = int(item)
    return vec


def readData(folderPath):
    '''
        读取文件夹folderPath下的所有图片数据，并将其存储为矩阵形式输出
    :param folderPath: 输入文件夹的地址
    :return: 矩阵输出
    '''
    files = os.listdir(folderPath)
    n = len(files)
    X = np.zeros((n, 1024))
    y = np.zeros((n))
    for index, imgName in enumerate(files):
        imgAbsPath = os.path.join(folderPath, imgName)
        X[index, :] = img2Vec(imgAbsPath)
        lable = imgName.split('.')[0].split('_')[0]
        y[index] = int(lable)
    print X.shape, y.shape
    return X, y


def classify(X_train, y_train, X_test, y_test, k):
    '''
        KNN分类器的代码实现，k为分类考虑的数目
    '''
    error = 0

    for index, X_line in enumerate(X_test):
        diff = np.tile(X_line, (X_train.shape[0], 1)) - X_train
        squ = diff ** 2
        sumSqu = np.sum(squ, axis=1)
        squ_root = sumSqu ** 0.5
        sortIndexList = np.argsort(squ_root)
        results = []
        for i in range(k):
            result = y_train[sortIndexList[i]]
            results.append(result)

        resultsCounter = Counter(results)
        lable = resultsCounter.most_common()[0][0]
        if y_test[index] != lable:
            error += 1
        print "{} is predict to {}".format(y_test[index], lable)
    errorRate = 1.0 * error / len(y_test)
    print "error rate is {}".format(errorRate)


def main():
    X_train, y_train = readData('trainingDigits')
    X_test, y_test = readData('testDigits')
    classify(X_train, y_train, X_test, y_test, 3)


if __name__ == "__main__":
    main()

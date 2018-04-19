from numpy import *
import operator
from os import listdir

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inx, dataSet, labels, k):
	# 计算距离
	# shape()函数返回数据集的大小，nxm，shape[0]=n,表示共有多少条数据，shape[1]=m表示每条数据的维度
	dataSetSize = dataSet.shape[0]

	#求欧氏距离
	# tile(A, reps)函数，对数组A进行复制。A=[1,2],tile(A,2)==>[1,2,1,2];tile(A,(1,2))==>[[1,2,1,2]]先进行x轴上复制2倍，再y轴上复制1倍
	diffMat = tile(inx, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	# sum(axis = 1)在矩阵的第1轴上进行相加，即求每一行的和，形成一个nx1的矩阵
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances**0.5

	#选择距离最小的k个点
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	#排序
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]


# # ---------约会预测----------
# 将文本记录转换为Numpy的解析程序
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	# 得到文件行数
	numberOfLines = len(arrayOLines)
	# 创建返回的numpy矩阵
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	# 解析文件数据到列表
	for line in arrayOLines:
		# strip()截取掉所有的回车字符
		line = line.strip()
		# 使用tab字符将上一步得到的整行数据分割成一个元素列表
		listFromLine = line.split('\t')
		# returnMat相当于前面的group
		returnMat[index,:] = listFromLine[0:3]
		# classLabelVector相当于前面的labels
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

# 归一化数值（同时对每一列进行操作）
def autoNorm(dataSet):
	# numpy的min()函数，min(0)表示选取矩阵每列中的最小值，min(1)表示选取每行中的最小值
	# minVals: 1x3
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	# 归一化后的矩阵:m x 3
	normDataSet = zeros(shape(dataSet))
	# m为行数，即数据条数
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet / tile(ranges, (m,1))
	return normDataSet, ranges, minVals

# 测试代码
def datingClassTest():
	# 设置测试集比例为0.1
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	# m为数据集大小（条数）
	m = normMat.shape[0]
	# 测试集选取数据集的10%
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
		if(classifierResult != datingLabels[i]): errorCount += 1.0
	print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	# (inArr - minVals)/ranges将inArr归一化
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print("You will probably like this person: ", resultList[classifierResult - 1])




# ---------手写数字识别----------
def img2vector(filename):
	"""将32x32的图像信息转换成1x1024的向量信息"""
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	"""将数据输入分类器，并检测分类器效果"""
	hwLabels = []
	# 获取目录内容
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		# 从文件名解析分类数字
		# 子文件名：0_0.txt、0_1.txt...
		fileNameStr = trainingFileList[i]
		# fileStr为0_0,0_1...
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		# 保存每个文件的数字类别
		hwLabels.append(classNumStr)
		# 保存相应向量
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
		if(classifierResult != classNumStr):
			errorCount += 1.0
		print("\nthe total number of errors is: %d" % errorCount)
		print("\nthe total error rate is: %f" % (errorCount/float(mTest)))



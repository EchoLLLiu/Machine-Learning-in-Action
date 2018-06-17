import numpy as np

def loadDataSet(filename):
	'''
	加载数据
	'''
	dataList =[]
	labelList = []
	with open(filename) as f:
		for line in f:
			lineArr = line.strip().split()
			dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
			labelList.append(int(lineArr[2]))
	return dataList, labelList

def sigmoid(z):
	'''
	sigmoid函数
	'''
	return 1.0/(1 + np.exp(-z))

def gradAscent(dataMatIn, classLabels, alpha = 0.001, MaxIter = 500):
	'''
	批处理梯度上升
	Args:
		dataMatIn: 训练集特征 
		classLabels：训练集分类标记
		alpha = 0.001：学习率
		MaxIter = 500：最大迭代次数
	Return:
		weights: 更新后的回归系数
    '''
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).T
	m, n = np.shape(dataMatrix)
	weights = np.ones((n,1))
	for k in range(MaxIter):
		h = sigmoid(dataMatrix * weights)
		error = labelMat - h
		weights = weights + alpha * dataMatrix.T * error
	return weights

def stocGradAscent0(dataList, labelList, alpha=0.01):
	'''
	随机梯度上升
	Args:
		dataMatrix: 训练集特征 
		classLabels：训练集分类标记
		alpha = 0.01：学习率
	Return:
		weights: 更新后的回归系数
	'''
	dataMatrix = np.mat(dataList)
	classLabels = np.mat(labelList).T
	m,n = np.shape(dataMatrix)
	weights = np.ones((n,1))
	for i in range(m):
		h = sigmoid(np.sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + dataMatrix[i].T * alpha * error 
	return weights

def stocGradAscent1(dataList, labelList, MaxIter=150):
	'''
	随机梯度上升(控制整个迭代次数以足够达到收敛，每次随机选取样本)
	Args:
		dataMatrix: 训练集特征 
		classLabels：训练集分类标记
		alpha = 0.01：学习率
	Return:
		weights: 更新后的回归系数
	'''
	dataMatrix = np.mat(dataList)
	classLabels = np.mat(labelList).T
	m,n = np.shape(dataMatrix)
	weights = np.ones((n,1))
	for j in range(MaxIter):
		dataIndex = range(m)
		for i in range(m):
			# 每次迭代体征学习率
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(np.random.uniform(0, len(dataIndex)))
			h = sigmoid(np.sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + dataMatrix[randIndex].T * alpha * error 
	return weights	

#--------------------------------------使用小例子进行测试------------------------------------
def classifyVector(inx, weights):
		'''
		分类函数
		Args:
			inx: 每个样本特征向量
			weights: 逻辑回归系数参数
		Return:
			分类标签1或0
		'''
		prob = sigmoid(np.sum(inx * weights))
		if prob > 0.5:
			return 1.0
		else:
			return 0.0

def colicTest():
	'''
	测试函数
	使用训练集训练模型，再使用测试集进行测试，并计算出错误率
	'''
	trainFile = 'data/horseColicTraining.txt'
	testFile = 'data/horseColicTest.txt'
	trainingSet = []
	trainingLabels = []
	# 训练出模型参数
	with open(trainFile) as ftrain:
		for line in ftrain:
			lineArr = line.strip().split('\t')
			lineArr = list(map(float, lineArr))
			trainingSet.append(lineArr[:-1])
			trainingLabels.append(lineArr[-1])
	trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
	# 使用训练好的模型进行验证
	errorCount = 0
	numTestVec = 0.0
	with open(testFile) as ftest:
		for line in ftest:
			numTestVec += 1.0
			lineArr = line.strip().split('\t')
			lineArr = list(map(float, lineArr))
			testArr = lineArr[:-1]
			testLabel = lineArr[-1]
			if int(classifyVector(testArr, trainWeights)) != int(testLabel):
				errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print('the error rate of this test is: %f' % errorRate)
	return errorRate

def multiTest():
	'''
	进行10次模型的训练与预测，并对10次的错误率求平均值
	'''
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print('after %d iterations the average error rate is : %f' % (numTests, errorSum/float(numTests)))
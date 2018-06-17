import numpy as np

def loadDataSet(filename, delim='\t'):
	'''
	加载数据集
	'''
	dataArr = []
	with open(filename) as fr:
		for line in fr:
			stringline = line.strip().split(delim)
			dataline = list(map(float, stringline))
			dataArr.append(dataline)
	return np.mat(dataArr)

def PCA(dataMat, topNfeat=9999999):
	# 去平均值
	meanVals = np.mean(dataMat, axis = 0)
	meanRemoved = dataMat - meanVals
	# np.cov()计算协方差，返回array
	covMat = np.cov(meanRemoved, rowvar=0)
	# 计算特征值与特征向量
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)
	# 最大的topNfeat个向量
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	# 新空间里降维后数据
	lowDataMat = meanRemoved * redEigVects
	print(np.shape(meanRemoved),np.shape(redEigVects),np.shape(lowDataMat))
	# 原空间里降维后的数据
	reconMat = (lowDataMat * redEigVects.T) + meanVals
	return lowDataMat, reconMat

#--------------------------------------使用小例子进行测试------------------------------------
def replaceNanWithMean(filename):
	'''将NaN数据用均值代替'''
	dataMat = loadDataSet(filename, ' ')
	numFeat = np.shape(dataMat)[1]
	for i in range(numFeat):
		# 计算每个特征的均值
		index = [i for i in range(np.shape(dataMat)[0])]
		NanIndex = np.nonzero(np.isnan(dataMat[:,i].A))[0]
		notNanIndex = [s for s in index if s not in NanIndex]
		meanVal = np.mean(dataMat[notNanIndex,i])
		dataMat[NanIndex, i] = meanVal
	return dataMat


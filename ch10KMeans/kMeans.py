import numpy as np

def loadDataSet(filename):
	'''
	加载数据集
	'''
	dataList = []
	with open(filename) as fr:
		for line in fr:
			curLine = line.strip().split('\t')
			fltLine = list(map(float, curLine))
			dataList.append(fltLine)
	return np.mat(dataList)

def distEclud(vecA, vecB):
	'''
	计算两个向量间的欧氏距离
	Args:
		vecA: 向量A
		vecB: 向量B
	Return:
		距离
	'''
	return np.sqrt(np.sum(np.power(vecA-vecB, 2)))

def randCent(dataSet, k):
	'''
	构建簇质心
	Args:
		dataSet: 数据集
		k: 簇个数
	Return:
		centroids: 簇质心
	'''
	n = np.shape(dataSet)[1]
	centroids = np.mat(np.zeros((k,n)))
	for j in range(n):
		minJ = np.min(dataSet[:,j])
		rangeJ = float(np.max(dataSet[:,j]) - minJ)
		# 保证随机生成的质心在整个数据集的边界之内
		centroids[:,j] = minJ + np.random.rand(k,1) * rangeJ  
	return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	'''
	k均值聚类算法
	Args:
		dataSet: 数据集
		k：需要聚类的簇的个数
		distMeas=distEclud：距离度量函数
		createCent=randCent：随机生成簇质心函数
	Return:
		centroids: 最终簇质心
		clusterAssment: 最终点聚类结果
	'''
	m = np.shape(dataSet)[0]
	# 第一个保存所属质心，第二个保存距离
	clusterAssment = np.mat(np.zeros((m,2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		# 将样本分配到相应的簇中
		for i in range(m):
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:], dataSet[i,:])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex, minDist ** 2
		print(centroids)
		# 更新簇质心
		for cent in range(k):
			ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
			centroids[cent,:] = np.mean(ptsInClust, axis=0)
	return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas = distEclud):
	'''
	二分k-均值聚类算法
	Args:
		dataSet: 数据集
		k：簇个数
		distMeas = distEclud：距离度量函数
	Return:
		centList: 簇质心
		clusterAssment: 聚类结果
	'''
	m = np.shape(dataSet)[0]
	clusterAssment = np.mat(np.zeros((m,2)))
	# 创建初始簇
	centroid0 = np.mean(dataSet, axis=0).tolist()[0]
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
	while len(centList) < k:
		lowestSSE = np.inf
		# 尝试划分每一簇
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0],:]
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
			# 划分后的误差平方和
			sseSplit = np.sum(splitClustAss[:,1])
			sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])
			print('sseSplit, and notSplit:',sseSplit, sseNotSplit)
			if sseSplit + sseNotSplit < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
		bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
		print('the bestCentToSplit is:', bestCentToSplit)
		print('the len of bestClustAss is:', len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0,:]
		centList.append(bestNewCents[1,:])
		clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss 
	return np.mat(np.array(centList)), clusterAssment


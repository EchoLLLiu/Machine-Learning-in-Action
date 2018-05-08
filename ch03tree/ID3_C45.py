# -*- coding:utf-8 -*-
from math import log
import operator

# C4.5算法与ID3算法仅有细微差别，其差别与代码在注释中体现
#-------------------------------------构造决策树-----------------------------------------
# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	# 为所有可能分类创建词典,key位分类标签（"属于鱼类""不属于鱼类"），value为每种分类标签个数
	labelCounts = {}
	for featVec in dataSet:
		# currentLabel保存分类标签
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannanEnt = 0.0
	# shannonEnt = -(求和)[p(xi)log(2,p(xi))]
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		# 以2为底求对数
		shannanEnt -= prob * log(prob,2)
	return shannanEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
	# 将dataSet中满足dataSet[axis] = value的行进行保留，retDataSet中存储了保留行中除了axis列之外的其它列
	retDataSet = []
	for featVec in dataSet:
		# 除掉featVec[axis]列的内容
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			# extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 遍历数据集，循环计算香农熵和splitDataSet()函数，找到最好的划分方式（计算所有特征的信息增益，并进行比较选出最优的特征）
def chooseBestFeatureToSplit(dataSet):
	# 计算总特征数，数据集最后一列为分类标签。dataSet[0]是指第一条数据
	numFeatures = len(dataSet[0]) - 1
	# 按照分类标签计算香农熵
	# baseEntropy为经验熵H(D)
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeaature = -1
	# 创建唯一的feature取值列表（分别对每个feature进行），对每个唯一feature值划分一次数据集
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		# newEntropy为经验条件熵H(D|A)
		newEntropy = 0.0
		# 计算每种划分方式的信息熵，并求该feature熵和
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			# subDataSet是feature[i]=value的所有条目的列表（不包含feature[i]）
			# len(subDataSet)表示feature[i]=value的条目总数
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
			# (C4.5)分裂信息：splitInfo
			# splitInfo -= prob * log(prob, 2)
		# 信息增益：g(D,A) = H(D) - H(D|A)
		inforGain = baseEntropy - newEntropy
		# (C4.5)信息增益率
		# inforGainRate = inforGain / splitInfo 
		if inforGain > bestInfoGain:
			bestInfoGain = inforGain
			bestFeaature = i
	return bestFeaature

# 当数据集处理了所有属性，但类标签依然不是唯一的时，采用“多数表决”的方法决定改叶子节点的分类
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount += 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

# 创建树的函数代码
def createTree(dataSet, labels):
	# 创建类别标签列表
	classList = [example[-1] for example in dataSet]
	# 类别完全相同则停止继续划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 遍历完所有特征时，返回出现次数最多的类别（dataset中只剩下一列类别）
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	# 得到列表包含的所有属性值
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

# 简单测试数据集
#def createDataSet():
#	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
#	labels = ['no surfacing','flippers']
#	return dataSet, labels

#-------------------------------------构建分类器-----------------------------------------
def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	# 将标签字符串转换为索引(featLabels与testVec中属性的顺序相同，即同特征具有同下标，但在featLabels中是属性名，testVec中是对应属性取值)
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel 

#-------------------------------------存储决策树（避免每次分类时都需要重新构建）-----------------------------------------
def storeTree(inputTree, filename):
	import pickle
	# fw = open(filename, "w")
	fw = open(filename, "wb+")
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr = open(filename, "rb+")
	return pickle.load(fr)
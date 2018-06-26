from numpy import *

def loadDataSet(fileName):
    '''数据加载函数'''
    dataArr = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float,curLine)) 
            dataArr.append(fltLine)
    return np.mat(dataArr)

def binSplitDataSet(dataSet, feature, value):
    '''特征切分函数
    Args:
        dataSet: 数据集
        feature: 切分特征
        value: 切分依据
    Return:
        mat0、mat1: 切分后的数据
    '''
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    '''叶节点生成函数
    Args:
        dataSet: 数据集
    Return:
        目标变量均值
    '''
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    '''误差估计函数
    Args:
        dataSet: 数据集
    Return:
        目标变量平方误差
    '''
    # np.var()计算的是均方差，需乘以样本数得总方差
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def linearSolve(dataSet):   
    '''格式化数据，便于进行线性回归'''
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    # 正规方程求解参数
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    #每个叶子节点是一个线性方程
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''找到最优的切分方式函数
    Args:
        dataSet: 数据集
        leafType: 叶节点类型（默认为回归树，也可以设定为模型树）
        errType:误差估计函数
        ops: 控制函数的停止时机
    Return:
        bestIndex: 最佳切分特征列
        bestValue: 最佳切分值
    '''
    # tolS是容许的误差下降值，tolN是切分的最少样本数
    tolS = ops[0]; tolN = ops[1]
    # 如果数据集中各特征值都一样，则为叶子节点
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    # 计算各特征值总方差
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    # 根据总方差来寻找到最佳的切分特征，以及切分依据值
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果划分样本数小于tolN，换个值继续尝试
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差下降小于阈值，不做切分，视为叶子节点
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) 
    # 如果按照此切分方式所得数据集很小，不做切分，视为叶子节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    
    return bestIndex,bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''树构建函数
    Args:
        dataSet: 数据集
        leafType: 叶节点类型（默认为回归树，也可以设定为模型树）
        errType:误差估计函数
        ops: 控制函数的停止时机
    Return:
        retTree构建好的树
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val 
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree 

# 回归树剪枝操作
def isTree(obj):
    '''判断是否是一棵树'''
    return (type(obj).__name__=='dict')

def getMean(tree):
    '''从上往下遍历树，直到叶节点，返回两节点均值'''
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            np.sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
# 测试哪种模型效果好
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, testData[i], modelEval)
    return yHat
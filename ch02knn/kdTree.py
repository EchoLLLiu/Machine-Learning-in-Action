# -*- coding: utf-8 -*-

#---------------------------------创建kd树----------------------------------
# kd-tree每个结点中主要包含的数据结构如下 
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split      # 整数（进行分割维度的序号）
        self.left = left        # 该结点分割超平面左子空间构成的kd-tree
        self.right = right      # 该结点分割超平面右子空间构成的kd-tree
 
 
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  # 数据维度
        
        def CreateNode(split, data_set): # 按第split维划分数据集exset创建KdNode
            if not data_set:    
                return None
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            # 找中位数
            split_pos = len(data_set) // 2      
            median = data_set[split_pos]        # 中位数分割点             
            split_next = (split + 1) % k        # cycle coordinates，下一次分割的维
            
            # 递归的创建kd树
            return KdNode(median, split, 
                          CreateNode(split_next, data_set[:split_pos]),     # 创建左子树
                          CreateNode(split_next, data_set[split_pos + 1:])) # 创建右子树
                                
        self.root = CreateNode(0, data)         # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历（根--左--右）
def preorder(root):  
    print(root.dom_elt)  
    if root.left:      # 节点不为空
        preorder(root.left)  
    if root.right:  
        preorder(root.right)  
      
#---------------------------------搜索kd树----------------------------------
from math import sqrt
from collections import namedtuple

# 定义一个nameduple，分别存放最近坐标点，最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point nearest_dist nodes_visited")

def find_nearest(tree, point):
    k = len(point)
    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)
        nodes_visited = 1
        s = kd_node.split          # 进行分割的维度
        pivot = kd_node.dom_elt    # 进行分割的节点
        # 找到离目标点最近的节点
        if target[s] <= pivot[s]:
            nearest_node = kd_node.left
            further_node = kd_node.right
        else:
            nearest_node = kd_node.right
            further_node = kd_node.left 
        # 进行遍历找到包含目标点的区域
        temp1 = travel(nearest_node, target, max_dist)
        # 以此叶节点作为“当前最近点”
        nearest = temp1.nearest_point
        # 更新最近距离
        dist = temp1.nearest_dist

        nodes_visited += temp1.nodes_visited

        # 最近点将在以目标点为球心，max_dist为半径的超球体内
        if dist < max_dist:
            max_dist = dist
        # 第s维上目标点与分割超平面的距离
        temp_dist = abs(pivot[s] - target[s])
        # 判断超球体是否与超平面相交
        # 如果不相交，则该点即为最近点
        if max_dist < temp_dist:
            return result(nearest, dist, nodes_visited)

        # 如果超球体与超平面相交，则在相交的区域中可能存在更近点
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1,p2 in zip(pivot, target)))

        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        # 检查另一个子节点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest, dist, nodes_visited)

    # 从根节点开始递归
    return travel(tree.root, point, float("inf"))

#---------------------------------测试结果----------------------------------

from time import clock
from random import random

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]
 
# 产生n个k维随机向量 
def random_points(k, n):
    return [random_point(k) for _ in range(n)]       
      
if __name__ == "__main__":
    data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]  # samples
    
    kd = KdTree(data)
    
    ret = find_nearest(kd, [3,4.5])
    print(ret)

    # N = 400000
    # kd2 = KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
    # t0 = clock()
    # ret2 = find_nearest(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
    # t1 = clock()
    # print("time: ",t1-t0, "s")
    # print(ret2)
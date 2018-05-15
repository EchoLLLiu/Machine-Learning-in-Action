# coding=utf-8

__author__ = "LY"
__time__ = "2018/5/11"

import bayes
import numpy as np

if '__main__' == __name__:
	listOPosts, ListClasses = bayes.loadDataSet()
	myVocabList =bayes.createVocabList(listOPosts)
	# print(myVocabList)
	# listOPosts0Vec = bayes.setOfWord2Vec(myVocabList, listOPosts[0])
	# listOPosts3Vec = bayes.setOfWord2Vec(myVocabList, listOPosts[3])
	# print(listOPosts0Vec)
	# print(listOPosts3Vec)

	# 使用小例子测试
	bayes.testingNB()
	
	# 进行垃圾邮件测试
	bayes.spamTest()


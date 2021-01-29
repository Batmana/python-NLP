# -*- coding: utf-8 -*-
import os, sys
import numpy as np 
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt 

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')
# 1. 导入数据
Input = file2matrix("testSet.txt","\t") # 1.导入数据
labels = Input[:,-1] # 获取分类标签列表
[m,n] = shape(Input) 

# 2. 构建b+x 系数矩阵：b这里默认为1
dataMat = buildMat(Input)
# 3. 定义步长和迭代次数 
alpha = 0.001  # 步长
steps = 500    # 迭代次数
weights = ones((n,1))  # 初始化权重向量
errorlist = []
# 4. 主程序
for k in xrange(steps):
	net = dataMat*mat(weights) # 待估计网络
	output = logistic(net)  # logistic函数
	loss = output-labels
	error = 0.5*sum(multiply(loss,loss))  # loss function
	errorlist.append(error)
	grad = dataMat.T*loss
	weights = weights - alpha*grad #梯度迭代

print weights	# 输出训练后的权重

# 5. 绘制训练后超平面
drawScatterbyLabel(plt,Input) # 2.按分类绘制散点图
X = np.linspace(-5,5,100)
Y=-(double(weights[0])+X*(double(weights[1])))/double(weights[2])
plt.plot(X,Y)
plt.show()


'''
# 6. 绘制训练后超平面
X = np.linspace(-5,5,100)
Ylist=[]
lenw = len(weightlist)
for indx in xrange(lenw):	
	if indx%20 == 0:   # 每20次输出一次分类超平面
		weight = weightlist[indx]
		Y=-(double(weight[0])+X*(double(weight[1])))/double(weight[2])
		plt.plot(X,Y)
		 #分类超平面注释
		plt.annotate("hplane:"+str(indx),xy = (X[99],Y[99]))
plt.show()		
'''
# -*- coding: UTF-8 -*-
# Filename : 03BPTest.py

from numpy import *
import BackPropgation
import Untils

# ���ݼ�
dataSet,classLabels = BackPropgation.loadDataSet("testSet2.txt") # ��ʼ��ʱ��1��Ϊȫ1����, studentTest.txt
dataSet = BackPropgation.normalize(mat(dataSet))

# �������ݵ�
# �ع�dataSet���ݼ�
dataMat = mat(ones((shape(dataSet)[0],shape(dataSet)[1])))
dataMat[:,1] = mat(dataSet)[:,0]
dataMat[:,2] = mat(dataSet)[:,1]	

# �������ݼ�ɢ��ͼ
Untils.drawClassScatter(dataMat, transpose(classLabels), False)

# BP������������ݷ���
errRec,WEX,wex = BackPropgation.bpNet(dataSet, classLabels)

# ����ͻ��Ʒ�����
x,z = BackPropgation.BPClassfier(-3.0, 3.0, WEX, wex)

Untils.classfyContour(x, x, z)

# �����������
X = linspace(0,2000,2000)
Y = log2(errRec)
Untils.TrendLine(X, Y)

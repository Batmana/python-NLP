# -*- coding: UTF-8 -*-
# Filename : 03BPTest.py

from numpy import *
import Untils
import BackPropgation

# ���ݼ�
dataSet = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
classLabels = [0,1,1,0]
expected = mat(classLabels)

# �������ݵ�
# �ع�dataSet���ݼ�
dataMat = mat(ones((shape(dataSet)[0],shape(dataSet)[1])))
dataMat[:,1] = mat(dataSet)[:,0]
dataMat[:,2] = mat(dataSet)[:,1]	

# �������ݼ�ɢ��ͼ
Untils.drawClassScatter(dataMat, transpose(expected), False)

# BP������������ݷ���
errRec,WEX,wex = BackPropgation.bpNet(dataSet, classLabels)

print errRec,WEX,wex

# ����ͻ��Ʒ�����
x = linspace(-0.2,1.2,30)
xx = mat(ones((30,30)))
xx[:,0:30] = x 
yy = xx.T
z = ones((len(xx),len(yy))) ;
for i in range(len(xx)):
   for j in range(len(yy)):
       xi = []; tauex=[] ; tautemp=[]
       mat(xi.append([xx[i,j],yy[i,j],1])) 
       hp = wex*(mat(xi).T)
       tau = BackPropgation.logistic(hp)
       taumrow,taucol= shape(tau)
       tauex = mat(ones((1,taumrow+1)))
       tauex[:,0:taumrow] = (tau.T)[:,0:taumrow]
       HM = WEX*(mat(tauex).T)
       out = BackPropgation.logistic(HM) 
       z[i,j] = out

Untils.classfyContour(x, x, z)

# �����������
X = linspace(0,1000,1000)
Y = log2(errRec)+1.0e-10
Untils.TrendLine(X, Y)


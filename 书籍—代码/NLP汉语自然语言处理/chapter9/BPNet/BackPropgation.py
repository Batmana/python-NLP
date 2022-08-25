# -*- coding: UTF-8 -*-
# Filename : BackPropgation.py
'''
Created on Oct 27, 2010
BP Working Module
@author: jack zheng
'''
from numpy import *
import Untils


# ���ݺ���:
def logistic(inX):
    return 1.0/(1.0+exp(-inX))

# ���ݺ����ĵ�����
def dlogit(inX1,inX2):
    return multiply(inX2,(1.0-inX2))

# �����Ԫ��ƽ��֮��
def errorfunc(inX):
    return sum(power(inX,2))/2.0
    
# ����student.txt���ݼ�
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename) #testSet.txt
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat   

# ���ݱ�׼��(��һ��):student.txt���ݼ�
def normalize(dataMat):
    # ��׼��
    dataMat[:,0] = (dataMat[:,0]-mean(dataMat[:,0]))/std(dataMat[:,0])
    dataMat[:,1] = (dataMat[:,1]-mean(dataMat[:,1]))/std(dataMat[:,1])
    return dataMat

def bpNet(dataSet,classLabels):
    # ���ݼ�����
    SampIn = mat(dataSet).T
    expected = mat(classLabels)
    m,n = shape(dataSet) 
    # �������
    eb = 0.01                   # ������� 
    eta = 0.05                  # ѧϰ�� 
    mc = 0.3                    # �������� 
    maxiter = 2000              # ���������� 
    errlist = []                # ����б�
    
    # ��������    
    # ��ʼ������
    nSampNum = m;    # ��������
    nSampDim = n-1;  # ����ά��
    nHidden = 4;   # ��������Ԫ 
    nOut = 1;      # �����
    
    # ���������
    hi_w = 2.0*(random.rand(nHidden,nSampDim)-0.5)  
    hi_b = 2.0*(random.rand(nHidden,1)-0.5) 
    hi_wb = mat(Untils.mergMatrix(mat(hi_w), mat(hi_b)))
    
    # ��������
    out_w = 2.0*(random.rand(nOut,nHidden)-0.5) 
    out_b = 2.0*(random.rand(nOut,1)-0.5)
    out_wb = mat(Untils.mergMatrix(mat(out_w), mat(out_b)))
    # Ĭ�Ͼ�Ȩֵ
    dout_wbOld = 0.0 ; dhi_wbOld = 0.0 

    for i in xrange(maxiter):   
        #1. �����ź����򴫲�
        
        #1.1 ����㵽������
        hi_input = hi_wb*SampIn
        hi_output = logistic(hi_input)        
        hi2out  = Untils.mergMatrix(hi_output.T, ones((nSampNum, 1))).T
        
    		#1.2 �����㵽�����    		
        out_input = out_wb*hi2out
        out_output = logistic(out_input)
        
        #2. ������     
        err = expected - out_output 
        sse = errorfunc(err)
        errlist.append(sse);
        #2.1 �ж��Ƿ�����
        if sse <= eb:
            print "iteration:",i+ 1    
            break;
        
        #3.����źŷ��򴫲�
        #3.1 DELTAΪ����㵽�������ݶ�  
        DELTA = multiply(err,dlogit(out_input,out_output))
        wDelta = out_wb[:,:-1].T*DELTA 
        
        #3.2 deltaΪ�����㵽������ݶ�
        delta = multiply(wDelta,dlogit(hi_input,hi_output))        
        dout_wb = DELTA*hi2out.T
        
        #3.3 ������Ȩֵ����
        dhi_wb = delta*SampIn.T    
        
        #3.4 ����������������Ȩֵ
        if i == 0:  
            out_wb = out_wb + eta * dout_wb 
            hi_wb = hi_wb + eta * dhi_wb
        else :    
            out_wb = out_wb + (1.0 - mc)*eta*dout_wb  + mc * dout_wbOld
            hi_wb = hi_wb + (1.0 - mc)*eta*dhi_wb + mc * dhi_wbOld
        dout_wbOld = dout_wb
        dhi_wbOld = dhi_wb     
    return errlist,out_wb,hi_wb

def BPClassfier(start,end,WEX,wex):
    x = linspace(start,end,30)
    xx = mat(ones((30,30)))
    xx[:,0:30] = x 
    yy = xx.T
    z = ones((len(xx),len(yy))) ;
    for i in range(len(xx)):
    	for j in range(len(yy)):
         xi = []; tauex=[] ; tautemp=[]
         mat(xi.append([xx[i,j],yy[i,j],1])) 
         hi_input = wex*(mat(xi).T)
         hi_out = logistic(hi_input) 
         taumrow,taucol= shape(hi_out)
         tauex = mat(ones((1,taumrow+1)))
         tauex[:,0:taumrow] = (hi_out.T)[:,0:taumrow]
         HM = WEX*(mat(tauex).T)
         out = logistic(HM) 
         z[i,j] = out 
    return x,z
             
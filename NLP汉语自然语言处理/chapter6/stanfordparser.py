# -*- coding: utf-8 -*-
import sys,os
from stanford import *

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['JAVA_HOME'] = 'D:\\Java7\\jdk1.8.0_65\\bin\\java.exe' # 配置环境变量

root = "E:/nltk_data/stanford-corenlp/" # 安装库
trainpath = "trainfile/" # chtb_0001.mrg
modelpath = "trainmodel.ser.gz"
txtmodelpath = "trainmodel.ser"
parser=StanfordParser(root)
parser.trainmodel(trainpath ,modelpath,txtmodelpath)
# result = parser.trainmodel(trainpath ,modelpath)





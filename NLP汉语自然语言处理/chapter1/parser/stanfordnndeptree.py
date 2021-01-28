# -*- coding: utf-8 -*-
import sys  
import os
from nltk.tree import Tree
from stanford import *

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 配置环境变量
os.environ['JAVA_HOME'] = 'D:\\Java7\\jdk1.8.0_65\\bin\\java.exe'
# 安装库

root = "E:/nltk_data/stanford-corenlp/"
modelpath= root+'models/lexparser/chinesePCFG.ser.gz'
opttype = 'typedDependencies' # "penn,typedDependencies"
parser=StanfordParser(modelpath,root,opttype)
result = parser.parse("罗马尼亚 的 首都 是 布加勒斯特 。")
print result




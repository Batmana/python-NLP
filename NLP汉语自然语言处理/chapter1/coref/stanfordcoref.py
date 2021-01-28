# -*- coding: utf-8 -*-
import sys  
import os
import nltk
from nltk.tree import Tree
from stanford import *

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 配置环境变量
os.environ['JAVA_HOME'] = 'D:\\Java7\\jdk1.8.0_65\\bin\\java.exe'
# 安装库
root = "E:/nltk_data/stanford-corenlp/"
modelpath= root+'models/hcoref/properties/zh-coref-default.properties'
coref = StanfordCoref(modelpath,root)
parsesent = '张国华是位老师，他为人热情，大家都叫他张老师。'
'''
(ROOT
  (IP
    (IP
      (NP (NN 张国华))
      (VP (VC 是)
        (NP
          (CLP (M 位))
          (NP (NN 老师)))))
    (PU ，)
    (IP
      (NP (PN 他))
      (VP (VV 为人)
        (NP (NN 热情))))
    (PU ，)
    (IP
      (NP (PN 大家))
      (VP
        (ADVP (AD 都))
        (VP (VV 叫)
          (NP
            (NP (PN 他))
            (NP (NR 张))
            (NP (NN 老师))))))
    (PU 。)))
'''
result = coref.annotate(parsesent)
print "result:",result




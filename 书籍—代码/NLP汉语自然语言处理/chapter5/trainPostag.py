# -*- coding: utf-8 -*-
import sys  
import os
from stanford import StanfordPOSTagger

reload(sys)  # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

# print os.environ
root = 'E:/nltk_data/stanford-corenlp/'
modelpath = root+"models/pos-tagger/chinese-distsim/chinese-distsim.tagger"
st = StanfordPOSTagger(root,modelpath)
propspath = "my.tagger.props"
st.trainmodel(propspath)

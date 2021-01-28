# -*- coding: utf-8 -*-
import sys  
import os
from framework import *
from stanford import StanfordNERTagger

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# print os.environ
root = 'E:/nltk_data/stanford-corenlp/'
modelpath = root+'models/ner/chinese.misc.distsim.crf.ser.gz'
st = StanfordNERTagger(modelpath,root)
seg_sent = '欧洲 东部 的 罗马尼亚 ， 首都 是 布加勒斯特 ， 也 是 一 座 世界性 的 城市 。 '
taglist = st.tagfile(seg_sent,"ner_test.txt")


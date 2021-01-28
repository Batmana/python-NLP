# -*- coding: utf-8 -*-
import sys,os
from pyltp import *
import re

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

words =  "张三 参加 了 这次 会议 。".split(" ")
postagger = Postagger()
postagger.load("E:\\nltk_data\\ltp3.3\\pos.model")
postags = postagger.postag(words)

parser = Parser()
parser.load("E:\\nltk_data\\ltp3.3\\parser.model")
arcs = parser.parse(words, postags)
arclen = len(arcs)
conll = ""
for i in xrange(arclen):
	if arcs[i].head ==0:
		arcs[i].relation = "ROOT"
	conll += str(i)+"\t"+words[i]+"\t"+postags[i]+"\t"+str(arcs[i].head-1)+"\t"+arcs[i].relation+"\n"	 
print conll

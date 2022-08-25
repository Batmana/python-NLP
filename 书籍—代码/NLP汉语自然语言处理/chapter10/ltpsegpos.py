# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import * 
# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

segmentor = Segmentor()
segmentor.load("E:\\nltk_data\\ltp3.3\\cws.model")
postagger = Postagger()
postagger.load("E:\\nltk_data\\ltp3.3\\pos.model")
text = "张三一大早就赶到了学校。他先到食堂吃早餐，然后他到宿舍拿自己的教材和笔记本。当他匆忙来到教室时，发现课本拿错了。"
words = segmentor.segment(text)
postags = postagger.postag(words)
for word,postag in zip(words,postags):
	print word+"/"+postag,

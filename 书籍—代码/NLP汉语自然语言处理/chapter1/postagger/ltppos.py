# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import * 
# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

sent = "在 包含 问题 的 所有 解 的 解空间树 中 ， 按照 深度优先 搜索 的 策略 ， 从 根结点 出发 深度 探索 解空间树 。"
words = sent.split(" ")

postagger = Postagger()
postagger.load("E:\\nltk_data\\ltp3.3\\pos.model")
postags = postagger.postag(words)
for word,postag in zip(words,postags):
	print word+"/"+postag,

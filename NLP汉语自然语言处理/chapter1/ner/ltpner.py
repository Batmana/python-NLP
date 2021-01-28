# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import * 

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

sent = "欧洲 东部 的 罗马尼亚 ， 首都 是 布加勒斯特 ， 也 是 一 座 世界性 的 城市 。"
words = sent.split(" ")
postagger = Postagger()
postagger.load("E:\\nltk_data\\ltp3.3\\pos.model")
postags = postagger.postag(words)

recognizer = NamedEntityRecognizer()
recognizer.load("E:\\nltk_data\\ltp3.3\\ner.model")
netags = recognizer.recognize(words, postags)

for word,postag,netag in zip(words,postags,netags):
	print word+"/"+postag+"/"+netag,
# -*- coding: utf-8 -*-
import sys
import os
from pyltp import Segmentor

reload(sys)
sys.setdefaultencoding('utf-8')

model_path = "E:\\nltk_data\\ltp3.3\\cws.model" #Ltp3.3 分词库
segmentor = Segmentor()
segmentor.load(model_path)

words = segmentor.segment("在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。")
print " | ".join(words) # 分割后的分词结果

# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import Segmentor
# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

model_path = "E:\\nltk_data\\ltp3.3\\cws.model"
user_dict = "E:\\nltk_data\\ltp3.3\\fulluserdict.txt" # 外部词典
segmentor = Segmentor()
segmentor.load_with_lexicon(model_path,user_dict)
sent = "在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。"
words = segmentor.segment(sent)
print " | ".join(words)

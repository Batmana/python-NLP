# -*- coding: utf-8 -*-
import sys  
import os
import jieba

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

jieba.load_userdict("userdict.txt")
sent = '在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。'
# 结巴分词--精确切分
wordlist = jieba.cut(sent)
print " | ".join(wordlist)
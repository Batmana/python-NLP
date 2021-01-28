# -*- coding: utf-8 -*-
import sys  
import os
import nltk
from nltk.tree import Tree
from nltk.grammar import DependencyGrammar
from nltk.parse import *
from pyltp import *
import re

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# words = "罗马尼亚 的 首都 是 布加勒斯特 。".split(" ")
words = "张三 参加 了 这次 会议".split(" ")
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
	conll += "\t"+words[i]+"("+postags[i]+")"+"\t"+postags[i]+"\t"+str(arcs[i].head)+"\t"+arcs[i].relation+"\n"	 
print conll
conlltree = DependencyGraph(conll)
tree = conlltree.tree()
tree.draw()

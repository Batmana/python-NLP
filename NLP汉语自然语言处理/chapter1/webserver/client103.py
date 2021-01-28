# -*- encoding: utf-8 -*-

import sys  
import os
import urllib
import time
import cPickle
import nltk
from nltk.tree import Tree
from nltk.grammar import DependencyGrammar
from nltk.parse import *

reload(sys)
sys.setdefaultencoding('utf-8')

# sent = "请输入测试句子"
sent = "尊敬的李慧龙，您的帐户已冻结，并转到法律催收。"
myurl = "http://192.168.1.103:8888/seg/"+sent
words = urllib.urlopen(myurl).read()
print words

myurl = "http://192.168.1.103:8888/pos/"+words
pos = urllib.urlopen(myurl).read()
print pos

myurl = "http://192.168.1.103:8888/parse/"+words+":"+pos
conll = urllib.urlopen(myurl).read()
print conll
conlltree = DependencyGraph(conll)
tree = conlltree.tree()
tree.draw()

myurl = "http://192.168.1.103:8888/stanfparse/"+str(words)
treedata = urllib.urlopen(myurl).read()
tree = cPickle.loads(treedata)
tree.draw()

# -*- encoding: utf-8 -*-

import sys  
import os
import urllib
import cPickle
import nltk
from nltk.tree import Tree
from nltk.grammar import DependencyGrammar
from nltk.parse import *

reload(sys)
sys.setdefaultencoding('utf-8')

sent = "尊敬的客户，卡尾号1234的信用卡05月未还金额1234.56元，请速缴清欠款，如已还款，请忽略本信息。[北京银行]。"
myurl = "http://localhost:8888/seg/"+sent
words = urllib.urlopen(myurl).read()
print words

myurl = "http://localhost:8888/pos/"+words
pos = urllib.urlopen(myurl).read()
print pos

myurl = "http://localhost:8888/parse/"+words+":"+pos
conll = urllib.urlopen(myurl).read()
print conll
conlltree = DependencyGraph(conll)
tree = conlltree.tree()
tree.draw()

myurl = "http://localhost:8888/stanfparse/"+" ".join(words)
treedata = urllib.urlopen(myurl).read()
tree = cPickle.loads(treedata)
tree.draw()

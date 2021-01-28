# -*- coding: utf-8 -*-
import sys  
import os
from pyltp import *

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

MODELDIR = "E:/nltk_data/ltp3.3/"
sentence = "欧洲东部的罗马尼亚，首都是布加勒斯特，也是一座世界性的城市。"
segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))
words = segmentor.segment(sentence)
wordlist = list(words)
postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))
postags = postagger.postag(words)

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model"))
arcs = parser.parse(words, postags)

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))
netags = recognizer.recognize(words, postags)

labeller = SementicRoleLabeller()
labeller.load(os.path.join(MODELDIR, "srl/"))
roles = labeller.label(words, postags, netags, arcs)

for role in roles:
	print 'rel:',wordlist[role.index]
	for arg in role.arguments:
		if arg.range.start!=arg.range.end:
			print arg.name,' '.join(wordlist[arg.range.start:arg.range.end])
		else:
			print arg.name,wordlist[arg.range.start]	
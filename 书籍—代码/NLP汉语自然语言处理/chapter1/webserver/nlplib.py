# -*- coding: utf-8 -*-

import sys  
import os
import nltk
from pyltp import *
from nltk.parse import stanford
'''
下载nltk的语料库
nltk.download()
'''
# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
def loadltpseg():
	model_path = "E:\\nltk_data\\cws.model"
	user_dict = "E:\\nltk_data\\userdict.txt"
	segmentor = Segmentor()
	segmentor.load_with_lexicon(model_path,user_dict)
	return segmentor
	
def loadltppos():
	postagger = Postagger()
	postagger.load("E:\\nltk_data\\pos.model")
	return postagger
	
def	loadparser():
	parser = Parser()
	parser.load("E:\\nltk_data\\parser.model")
	return parser
	
# 配置环境变量
os.environ['JAVAHOME'] = 'D:\\Java7\\jdk1.8.0_51\\bin\\java.exe'
os.environ['STANFORD_PARSER'] = 'E:\\nltk_data\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'E:\\nltk_data\\stanford-parser-3.5.0-models.jar'
def loadstanford():
	return stanford.StanfordParser(model_path="E:\\nltk_data\\chinesePCFG.ser.gz")
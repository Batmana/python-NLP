# -*- coding: utf-8 -*-
import sys  
import os
# CoreNLP 3.6 jar包和中文模型包
# ejml-0.23.jar
# javax.json.jar
# jollyday.jar
# joda-time.jar
# jollyday.jar
# protobuf.jar
# slf4j-api.jar
# slf4j-simple.jar
# stanford-corenlp-3.6.0.jar
# xom.jar
class StanfordCoreNLP(object):
	def __init__(self,jarpath):
		self.root = jarpath
		self.tempsrcpath = "tempsrc" # 输入临时文件路径		
		self.jarlist = ["ejml-0.23.jar","javax.json.jar","jollyday.jar","joda-time.jar","protobuf.jar","slf4j-api.jar","slf4j-simple.jar","stanford-corenlp-3.6.0.jar","xom.jar"]
		self.jarpath = ""
		self.buildjars()
		
	def buildjars(self):	# 根据root路径构建所有的jar包路径
		for jar in self.jarlist: 
			self.jarpath += self.root+jar+";"
			
	def savefile(self,path,sent):
		fp = open(path,"wb")
		fp.write(sent)
		fp.close()	
	# 读取和删除临时文件
	def delfile(self,path):
		os.remove(path)		
		
class StanfordParser(StanfordCoreNLP):
	def __init__(self,jarpath):
		StanfordCoreNLP.__init__(self,jarpath)
		self.modelpath = "" # 模型文件路径
		self.classfier = "edu.stanford.nlp.parser.lexparser.LexicalizedParser"
		self.opttype = ""
	# 构建命令行	
	def __buildcmd(self):
		self.cmdline = 'java -mx500m -cp "'+self.jarpath+'" '+self.classfier+' -outputFormat "'+self.opttype+'" '+self.modelpath+' '
	#解析句子
	def parse(self,sent):
		self.savefile(self.tempsrcpath,sent)
		tagtxt = os.popen(self.cmdline+self.tempsrcpath,"r").read()	# 输出到变量中
		self.delfile(self.tempsrcpath)
		return 	tagtxt
	# 输出到文件	
	def tagfile(self,sent,outpath):
		self.savefile(self.tempsrcpath,sent)
		os.system(self.cmdline+self.tempsrcpath+' > '+outpath )		
		self.delfile(self.tempsrcpath)	
	
	def __buildtrain(self,trainpath,parsemodel): # 创建模型文件	
		self.trainline = 'java -mx2g -cp "'+self.jarpath+'" '+self.classfier +' -tLPP edu.stanford.nlp.parser.lexparser.ChineseTreebankParserParams -train "'+trainpath+'" -saveToSerializedFile "'+parsemodel+'"'		

	def __buildtraintxt(self,trainpath,parsemodel,txtmodel): # 创建模型文件	
		self.trainline = 'java -mx2g -cp "'+self.jarpath+'" '+self.classfier +' -tLPP edu.stanford.nlp.parser.lexparser.ChineseTreebankParserParams -train "'+trainpath+'" -saveToSerializedFile "'+parsemodel+'" -saveToTextFile "'+txtmodel+'"'

	def trainmodel(self,trainpath,parsemodel,txtmodel=""): # 训练模型
		if txtmodel:
			self.__buildtraintxt(trainpath,parsemodel,txtmodel)
		else:	
			self.__buildtrain(trainpath,parsemodel)
		os.system(self.trainline)	
		print "save model to ",parsemodel
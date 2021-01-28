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
		
class StanfordPOSTagger(StanfordCoreNLP):
	def __init__(self,jarpath,modelpath):
		StanfordCoreNLP.__init__(self,jarpath)
		self.modelpath = modelpath # 模型文件路径
		self.classfier = "edu.stanford.nlp.tagger.maxent.MaxentTagger"
		self.delimiter = "/"
		self.__buildcmd()
		
	def __buildcmd(self):	# 构建命令行	
		self.cmdline = 'java -mx1g -cp "'+self.jarpath+'" '+self.classfier+' -model "'+self.modelpath+'" -tagSeparator '+self.delimiter
		
	def tag(self,sent):	#标注句子
		self.savefile(self.tempsrcpath,sent)
		tagtxt = os.popen(self.cmdline+" -textFile "+self.tempsrcpath,"r").read()	# 执行命令行
		self.delfile(self.tempsrcpath)
		return 	tagtxt
	def tagfile(self,inputpath,outpath):# 标注文件
		self.savefile(self.tempsrcpath,sent)
		os.system(self.cmdline+' -textFile '+self.tempsrcpath+' > '+outpath )		
		self.delfile(self.tempsrcpath)
		
		
class StanfordNERTagger(StanfordCoreNLP):
	def __init__(self,modelpath,jarpath):
		StanfordCoreNLP.__init__(self,jarpath)
		self.modelpath = modelpath # 模型文件路径
		self.classfier = "edu.stanford.nlp.ie.crf.CRFClassifier"
		self.__buildcmd()
	# 构建命令行	
	def __buildcmd(self):
		self.cmdline = 'java -mx1g -cp "'+self.jarpath+'" '+self.classfier+' -loadClassifier "'+self.modelpath+'"'
	#标注句子
	def tag(self,sent):
		self.savefile(self.tempsrcpath,sent)
		tagtxt = os.popen(self.cmdline+' -textFile '+self.tempsrcpath,'r').read()	# 执行命令行
		self.delfile(self.tempsrcpath)
		return 	tagtxt
	# 标注文件	
	def tagfile(self,sent,outpath):
		self.savefile(self.tempsrcpath,sent)
		os.system(self.cmdline+' -textFile '+self.tempsrcpath+' > '+outpath )		
		self.delfile(self.tempsrcpath)

class StanfordParser(StanfordCoreNLP):
	def __init__(self,modelpath,jarpath,opttype):
		StanfordCoreNLP.__init__(self,jarpath)
		self.modelpath = modelpath # 模型文件路径
		self.classfier = "edu.stanford.nlp.parser.lexparser.LexicalizedParser"
		self.opttype = opttype
		self.__buildcmd()
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
		
		
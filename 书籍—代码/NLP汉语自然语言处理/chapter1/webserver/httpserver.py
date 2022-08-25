# -*- coding: utf-8 -*-
import sys  
import os
import nltk
import cPickle
from nltk.tree import Tree
from nltk.parse import *
from pyltp import *
import tornado.httpserver
import tornado.ioloop
import tornado.web
from nlplib import *
from nltk.parse import stanford

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')


segmentor = loadltpseg()
postagger = loadltppos()
parser = loadparser()
# 安装库
stanfparser=loadstanford()

class MainHandler(tornado.web.RequestHandler):
		def get(self):
			self.write("Hello, world")
			
class SegHandler(tornado.web.RequestHandler):
		def get(self, input_sent):
			wordlist = segmentor.segment(str(input_sent))
			self.write(" ".join(wordlist))

class PosHandler(tornado.web.RequestHandler):
		def get(self, input_sent):
			wordlist = str(input_sent).split(" ")
			taglist = postagger.postag(wordlist)
			self.write(" ".join(taglist))
			
class ParseHandler(tornado.web.RequestHandler):
		def get(self, input_sent):
			elelist = str(input_sent).split(":")
			wordlist = str(elelist[0]).split(" ")
			postags = str(elelist[1]).split(" ")
			arcs = parser.parse(wordlist,postags)
			arclen = len(arcs)
			conll = ""
			for i in xrange(arclen):
				if arcs[i].head ==0:
					arcs[i].relation = "ROOT"
				conll += "\t"+wordlist[i]+"("+postags[i]+")"+"\t"+postags[i]+"\t"+str(arcs[i].head)+"\t"+arcs[i].relation+"\n"	 			
			self.write(conll)
			
class Stanfordparser(tornado.web.RequestHandler):
		def get(self, input_sent):	
			sentiter = stanfparser.raw_parse(str(input_sent))
			tree = Tree('',sentiter)
			treestr = cPickle.dumps(tree)
			self.write(treestr)
						
application = tornado.web.Application([
		(r"/", MainHandler),
		(r"/seg/(.*)", SegHandler),
		(r"/pos/(.*)", PosHandler),
		(r"/parse/(.*)", ParseHandler),
		(r"/stanfparse/(.*)", Stanfordparser),		
])

if __name__ == "__main__":
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(8888)
 	tornado.ioloop.IOLoop.instance().start()
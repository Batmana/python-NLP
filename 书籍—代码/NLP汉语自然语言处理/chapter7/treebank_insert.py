# -*- coding: utf-8 -*-
import sys  
import os
import re
import traceback
from framework import *
from treelib import *
from nltk.tree import Tree,ParentedTree
from mySQL import CMySql 
from MySQLdb import * 

# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 将树库和文件名输出到一个文件
rootdir = "treebanks/ctb8.0/data/bracketed/"
DBconn =  CMySql("127.0.0.1", "root", "root", "treebanks",3306)
filelist = os.listdir(rootdir)
count = 0
for filename in filelist:
	linelist = readfile(rootdir+filename).splitlines()
	linelen = len(linelist)
	treelist = []
	begin = end = -1; indx=0
	while indx < linelen:
		if (begin == -1 ) and linelist[indx] and linelist[indx][0]=="(":
			begin = indx
		elif (begin != -1 ) and ( linelist[indx]==""  or linelist[indx][0]=="<" or linelist[indx][0]=="(" or indx==(linelen-1) ): 
			end = indx
			if indx==(linelen-1) and linelist[indx] and linelist[indx][0]!="(" : 
				treelist.append("\n".join(linelist[begin:]))
			else:
				treelist.append("\n".join(linelist[begin:end]))
			if linelist[indx] and linelist[indx][0]=="(" : indx -= 1	
			begin = end = -1
		indx +=1	
	file_seq = 0;
	for treestr in treelist :
		fieldlist =[]
		sentTree = escape_string(treestr)
		flatTree = escape_string(treestr.replace("\n"," ").replace("\t"," "))
		try:
			mytree = Tree.fromstring(treestr)
			wordlist = mytree.leaves()
			sentSegment = escape_string(" ".join(wordlist))
			sentence = escape_string("".join(wordlist))
			sentPOS = escape_string(" ".join([word_pos[0][0]+"/"+word_pos[0][1] for word_pos in flatten_deeptree(mytree).pos()]))
		except Exception,te:
			print filename,flatTree
			print te
			sys.exit()
		file_seq +=1
		fieldlist.append(escape_string(filename));fieldlist.append(str(file_seq));
		fieldlist.append(sentence);fieldlist.append(sentSegment);fieldlist.append(sentPOS)
		fieldlist.append(sentTree);fieldlist.append(flatTree);
		# 插入数据库的语句：
		insertsql = "INSERT INTO penn_treebank VALUES('','"+"','".join(fieldlist)+"')"
		count +=1
		# DBconn.insert(insertsql)
print count,"ok"
# Id,filename,file_seq,sentence,sentsegment,sentPOS,sentTree,flatTree	
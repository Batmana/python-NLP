# -*- coding: utf-8 -*-
import sys  
import os
import nltk
from framework import *
from treelib import *
from mySQL import CMySql 
from MySQLdb import *
from nltk.tree import Tree
'''
下载nltk的语料库
nltk.download()
'''
ptr = re.compile(r"[0-9]+")
# 设置 UTF-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')
# 关联verb的treeid
root = "E:\\CTB_Corpus\\treebanks\\cpb3.0\\data\\"
verbdir = root+"cpb3.0-verbs.txt"
verblist = readfile(verbdir).splitlines()
print len(verblist)
DBconn =  CMySql("127.0.0.1", "root", "root", "testbank",3306)
# Id,filename,seq,treeid,predicatePOS,rel,srlstr,augments
for prop in verblist:
	proplist = parseProp(prop,"verb",DBconn)
	sql = "INSERT INTO propbank VALUES('','"+"','".join(proplist)+"')"
	DBconn.execute(sql)


print "ok"
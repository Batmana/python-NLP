# -*- coding: utf-8 -*-

import sys  
import os
import traceback  
import time
from framework import *
import nltk
import re
from cStringIO import StringIO

reload(sys)   
sys.setdefaultencoding('utf8') 

def fullfreq(fcontent,sumdict):
	sent = " ".join(fcontent.splitlines()).strip().decode("utf8")
	sTuple=[ nltk.tag.str2tuple(t) for t in sent.split(" ") ]
	fredist=nltk.FreqDist(sTuple) #获取统计结果
	print len(fredist)
	for localkey in fredist:
		if localkey in sumdict : #检查当前词频是否在字典中存在
			sumdict[localkey]=sumdict[localkey]+fredist[localkey] #如果存在，将词频累加，并更新字典值
		elif str(localkey[1]).find("None")==-1 :
			sumdict[localkey]=fredist[localkey] #将当前词频添加到字典中	

hanzi= re.compile(ur"[\u4e00-\u9fa5]+") #切分汉字
def hanzfreq(fcontent,sumdict):
	sent = " ".join(fcontent.splitlines()).strip().decode("utf8")
	sTuple=[ nltk.tag.str2tuple(t) for t in sent.split(" ") if hanzi.match(t) ]
	fredist=nltk.FreqDist(sTuple) #获取统计结果
	print len(fredist)
	for localkey in fredist:
		if localkey in sumdict : #检查当前词频是否在字典中存在
			sumdict[localkey]=sumdict[localkey]+fredist[localkey] #如果存在，将词频累加，并更新字典值
		elif str(localkey[1]).find("None")==-1 : 
			sumdict[localkey]=fredist[localkey] #将当前词频添加到字典中		

sumdict={}	# 统计结果
rootdir = "E:/nltk_data/segments/"
segpath = "swresult1998.txt"
segcorpus = readfile(rootdir+segpath)
# fullfreq(segcorpus,sumdict) # 包含非汉字的词频统计
hanzfreq(segcorpus,sumdict) # 仅有汉字的词频统计
sumlist= sorted(sumdict.items(), key=lambda x:x[1],reverse=True)
file_str = StringIO()		
for key in sumlist:
	file_str.write(str(key[0][0]));	file_str.write("\t")
	file_str.write(str(key[0][1]));	file_str.write("\t")	
	file_str.write(str(key[1]));	file_str.write("\n")
savefile(rootdir+"freqdict.txt",file_str.getvalue()) #freqdict.txt
print "ok"
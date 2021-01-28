# -*- coding: utf-8 -*-

import sys  
import os
import time
import random
import time
import re
from urllib2 import *

# 文件处理函数库
# 保存至文件
def savefile(savepath,content):
	fp = open(savepath,"wb")
	fp.write(content)
	fp.close()

# 附加至文件
def appendfile(savepath,content):
	fp = open(savepath,"a")
	fp.write(content)
	fp.close()
	
# 批量保存至文件
def dumpDic2file(prefix,mydict):
	for myfilename in mydict:
		fp = open(prefix+myfilename,"wb")
		fp.write(mydict[myfilename])
		fp.close()
# 批量保存至文件:文件名为unicode
def dumpDicUnicode(prefix,mydict):
	for myfilename in mydict:
		fp = open(unicode(prefix+myfilename),"wb")
		fp.write(mydict[myfilename])
		fp.close()
# 读取文件
def readfile(path):
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	return content
	
# 读取文件行
def readfilelines(path,nmax=0):
	flag = True
	fp = open(path,"rb")
	ncount =0
	while flag:
		content = fp.readline()
		if content =="" or (ncount>nmax):
			flag = False
		yield content
		if nmax != 0 : 
			ncount += 1
	fp.close()	
	
# 从目录中读取全部文件内容
def read_from_dir(mydir):
	files = os.listdir(mydir)  # 获取mydir下的所有文件 
	pages = [readfile(mydir+htmlfile) for htmlfile in  files]
	return pages,files

# 从目录中读取全部文件内容并返回dict
def read2dict(mydir,mydict):
	files = os.listdir(mydir)  # 获取mydir下的所有文件
	for htmlfile in files:
		mydict[htmlfile]=readfile(mydir+htmlfile) 
	return mydict
		
# 存储为日志文件
def writelog(loginfo):
	logfile = "F:\\log.txt"
	fp = open(logfile,"a")
	fp.write(loginfo)					
	fp.close()			
	
# 转为xml路径和名称		
def rename2xml(htmfilename,htmlpath,xmlpath):
	xmlfilename = htmfilename.replace(".htm",".xml")
	xmlfilename = xmlfilename.replace(htmlpath,xmlpath)
	return xmlfilename

# 递归创建与srcprefix相同的目录结构
def duplicatedirs(srcprefix,destprefix):
	srcdirlist = os.listdir(srcprefix)
	for srcdir in srcdirlist:
		level1path = srcprefix+srcdir			
		if os.path.isdir(level1path):
			newdir = destprefix+srcdir
			if not os.path.exists(newdir):
				os.mkdir(newdir)
			duplicatedirs(level1path+"\\",newdir+"\\")

# 从根目录输出文件路径列表
def doclistFile(rootdir,savepath):
	dirlist = os.listdir(rootdir)
	filelist = []
	for mydir in dirlist:
		sublist = os.listdir(rootdir+mydir)
		for mypath in sublist:
			fullpath = rootdir+mydir+"\\"+mypath
			if os.path.isdir(fullpath):
				ambipathlist = os.listdir(fullpath)
				for ambipath in ambipathlist:
					filelist.append(fullpath+"\\"+ambipath)
			else:
				filelist.append(fullpath)
	savefile(savepath,"\n".join(filelist))
	
# 比较两个htm和xml的目录的差异
def htm2xmldiff(htmldirspath,xmldirspath,htmprefix,xmlprefix):
	fullset = set()
	htmldirs = set()
	xmlist = readfile(xmldirspath).splitlines()
	set(fullset.add(xmlfulldir.strip().replace(xmlprefix,htmprefix).replace(".xml",".htm")) for xmlfulldir in xmlist)
	htmlist = readfile(htmldirspath).splitlines()
	set(htmldirs.add(htmdir.strip()) for htmdir in htmlist)
	return htmldirs.difference(fullset)

# 生成文件目录
def pathstr2dir(pathnum):
	len_pn = len(pathnum)	
	if len_pn <= 5:
		return  "00000000_00100000\\"
	if len_pn == 6:
		intpn = int(pathnum[0])
		if intpn != 9:
			return  "00"+pathnum[0]+"00000_00"+str(intpn+1)+"00000\\"
		else:
			return  "00"+pathnum[0]+"00000_0"+str(intpn+1)+"00000\\"			
	elif len_pn ==7:
		intfn = int(pathnum[:2])
		if len(str(intfn+1))==3:
			return  "0"+pathnum[:2]+"00000_"+str(intfn+1)+"00000\\"
		else:
			return  "0"+pathnum[:2]+"00000_0"+str(intfn+1)+"00000\\"
	elif len_pn ==8:
		intfn = int(pathnum[:3])
		return  pathnum[:3]+"00000_"+str(intfn+1)+"00000\\"
			

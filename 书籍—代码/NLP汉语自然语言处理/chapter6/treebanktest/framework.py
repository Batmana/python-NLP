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
		content = fp.readline().strip()
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

# 从目录中读取全部文件内容
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
			
##过滤HTML中的标签
#将HTML中标签等信息去掉
#@param htmlstr HTML字符串.
def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    # s=replaceCharEntity(s)#替换实体
    return s

##替换常用HTML字符实体.
#使用正常的字符替换HTML中特殊的字符实体.
#你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
#@param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
                'lt':'<','60':'<',
                'gt':'>','62':'>',
                'amp':'&','38':'&',
                'quot':'"','34':'"',}
   
    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        entity=sz.group()#entity全称，如&gt;
        key=sz.group('name')#去除&;后entity,如&gt;为gt
        try:
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            #以空串代替
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
    return htmlstr
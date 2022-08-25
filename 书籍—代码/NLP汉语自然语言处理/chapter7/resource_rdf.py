# -*- coding: utf-8 -*-

import sys  
import os
import re
import time
from framework import *
from MySQLdb import *
from mySQL import CMySql 
import rdflib

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

class RdfDB(object):
	# 构造方法
	def __init__(self, table_prefix=""):
		self.tablename = ""
		self.tablelist = []
		self.triplelist = []
		self.DBconn = ""
		self.resource = ""
		self.dbname = ""
		self.table_prefix = table_prefix if table_prefix != "" else "resource_"
	
	# 获取所有表名
	def get_tablelist(self):
		tablesql = "show tables;"
		key = 'Tables_in_'+ self.dbname 
		if self.DBconn != "":
			results = self.DBconn.query(tablesql)
			for result in results:
				if result[key].find(self.table_prefix) != -1:
					self.tablelist.append(result[key]) 
	#从表列表中获取资源				
	def triplefromtablelist(self):
		for tablename in self.tablelist:
			resource_sql = "SELECT * FROM "+escape_string(tablename)+" WHERE resource = '"+escape_string(self.resource)+"'"
			results = self.DBconn.querylist(resource_sql)
			for result in results:
				print list(result)[1]
				print list(result)[2]
				print list(result)[3]
				#self.triplelist.append()
				
	#从表列表中获取资源				
	def get_triplelist(self,tablename):
		resource_sql = "SELECT * FROM "+escape_string(tablename)+" WHERE resource = '"+escape_string(self.resource)+"'"
		results = self.DBconn.query(resource_sql)
		for result in results:
			print (list(result))[1],
			print (list(result))[2],
			print (list(result))[3]
			#self.triplelist.append()
	
	def make_resouce(self,prefix,keyword):
		self.resource = prefix[:-1]+keyword+">"		
	
	def build_Conn(self,host,pwd,usr,db,port):
		self.DBconn =  CMySql(host, pwd, usr, db,port)
		self.dbname = db	
		
rdf = RdfDB()
rdf.build_Conn("127.0.0.1", "root", "root", "wiki_pedia",3306)
rdf.get_tablelist()
rdf.make_resouce("<http://zh.dbpedia.org/resource/>","孫中山")
rdf.get_triplelist("resource_category")
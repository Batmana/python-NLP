#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

try:
    import MySQLdb
except ImportError:
    raise ImportError("[E]: MySQLdb module not found!")

class CMySql(object):
	def __init__(self,host, pwd, user, db, port=3306):
		self.Option = {"host" : host, "password" : pwd, "username" : user, "database" : db,"port":port}
		#self.__conn = 0
		#self.cursor = 0
		#self.__dictcursor = 0
		self.__connection__()
	
	def __del__(self):
		if self.__conn:
			 self.close()
	
	def __connection__(self):
		try:
			self.__conn = MySQLdb.connect(
                        host = self.Option["host"], user = self.Option["username"],
                        passwd = self.Option["password"], db = self.Option["database"],
                        port = self.Option["port"],charset='utf8')
			self.__dictcursor = MySQLdb.cursors.DictCursor   
		except Exception, e:
			print e
			raise Exception("[E] Cannot connect to %s" % self.Option["host"])
	
	def execute_once(self, sqlstate):
		# @todo: 增、删、改--关闭数据库
		self.cursor = self.__conn.cursor()
		self.cursor.execute(sqlstate) 
		self.commit()  # 执行事物
		self.cursor.close()

	def execute(self, sqlstate):
		# @todo: 增、删、改--不关闭数据库
		self.cursor = self.__conn.cursor()
		self.cursor.execute(sqlstate) 
		self.commit()  # 执行事物
		
	def insert(self, sqlstate):
		# @todo: 增、删、改--不关闭数据库
		self.cursor = self.__conn.cursor()
		self.cursor.execute(sqlstate) 
		lastinsertid = int(self.__conn.insert_id())
		self.commit()  # 执行事物
		return lastinsertid
				 
	# 关闭数据库           
	def query_once(self, sqlstate):
		self.cursor = self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate) #查询
		qres = self.cursor.fetchall()
		self.cursor.close()
		return qres
	# 关闭数据库       
	def queryone_once(self, sqlstate):
		self.cursor = self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate) #查询
		qres = self.cursor.fetchone()
		self.cursor.close()
		return qres
	# 不关闭数据库           
	def query(self, sqlstate):
		self.cursor = self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate) #查询
		return self.cursor.fetchall()
	
	def querylist(self, sqlstate,size):
		self.cursor = self.__conn.cursor()
		self.cursor.execute(sqlstate) #查询
		return self.cursor.fetchmany(size)
		
	# 不关闭数据库       
	def queryone(self, sqlstate):
		self.cursor = self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate) #查询
		return self.cursor.fetchone()
		
	def close(self):
		self.__conn.close()
		
	def commit(self):
		try:
			self.__conn.commit()
		except:
			self.__conn.rollback()
			raise

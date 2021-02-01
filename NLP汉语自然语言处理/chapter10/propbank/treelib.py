# -*- coding: utf-8 -*-
import sys, os
import re
import nltk
from nltk.tree import *
from framework import *
from MySQLdb import *
from nltk.corpus import propbank

def flatten_childtrees(trees): # 扁平化子树结构
	children = []
	for t in trees:
		if t.height() < 3:
			children.extend(t.pos())
		elif t.height() == 3:
			children.append(Tree(t.label(), t.pos()))
		else:
			children.extend(flatten_childtrees([c for c in t]))
	return children

def flatten_deeptree(tree): # 解析prop	
	return Tree(tree.label(), flatten_childtrees([c for c in tree]))
	
def getwordposlist(tree):
	return [ tree[pos]+"/"+tree[pos[:-1]].label() for pos in tree.treepositions('leaves')]

def getbranch(tree,keyword,branchlabel):
	gspos = tuple()
	for pos in tree.treepositions('leaves'):
		if tree[pos]==keyword: gspos = pos 
	indx = -1
	for count in xrange(len(gspos)-1):
		if tree[gspos[:indx]].label()==branchlabel:
			return tree[gspos[:indx]]
		indx -= 1
		
# 解析prop	
def parseProp(propstr,postag,DBConn):
	proplist = []
	prefix = propstr.split("-----")
	term = prefix[0].split(" ")
	filename = term[0]
	seq = str(int(term[1])+1)
	predicatePOS = postag
	rel = term[4]
	srlstr = propstr
	sql = "SELECT * FROM treebank WHERE filename = '"+filename+"' AND seq='"+seq+"'"
	record = DBConn.query(sql)
	treeid = str(record[0]['Id'])
	flattree = record[0]['flatTree']
	augments = get_augments(flattree,srlstr,treeid)
	proplist.append(escape_string(filename)); proplist.append(seq);
	proplist.append(treeid);proplist.append(predicatePOS);
	proplist.append(escape_string(rel));proplist.append(escape_string(srlstr));
	proplist.append(escape_string(augments))
	return proplist
	
def get_augments(treestr,srlstr,treeid):
	augments = ""
	tree = Tree.fromstring(treestr)
	try:
		inst =  propbank.read_instance(srlstr)
		treepos = inst.predicate.treepos(tree)
		relstr = repr(tree[treepos]).replace("\n","")+"\n"
		augments += "rel: " + relstr.decode("unicode-escape")
		auglist =[ str(argid)+": "+ repr(argloc.select(tree)).decode("unicode-escape").replace("\n","") for (argloc, argid) in inst.arguments ]
		if auglist:  augments += u"\n".join(auglist)
	except Exception,e:
		print e 
		appendfile("emptytree.txt",srlstr+"-->treeid:"+str(treeid)+"\n")
		# sys.exit()
	return augments				
# -*- coding: utf-8 -*-
import sys, os
import re
import nltk
from nltk.tree import *
from framework import *

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
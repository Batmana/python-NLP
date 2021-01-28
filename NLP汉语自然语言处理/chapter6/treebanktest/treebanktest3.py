# -*- coding: utf-8 -*-
import sys, os
import re
from treelib import *

reload(sys)  # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')
mytree = Tree.fromstring(u"(IP (IP (NP-SBJ (NN 建筑)  (NN 公司))  (VP (VV 进)    (NP-OBJ (NN 区)))) (PU ，)  (IP (NP-SBJ (ADJP (JJ 有关))  (NP (NN 部门)))  (VP (ADVP (AD 先))     (VP (VV 送上)  (NP-OBJ (DP (DT 这些))       (NP (NN 法规性)  (NN 文件))))))   (PU ，)   (IP (NP-SBJ (-NONE- *pro*))   (VP (ADVP (AD 然后))         (VP (VE 有)  (IP-OBJ (NP-SBJ (ADJP (JJ 专门))  (NP (NN 队伍)))   (VP (VV 进行)     (NP-OBJ (NN 监督)     (NN 检查)))))))  (PU 。))  ")
ptree = ParentedTree.convert(mytree)
# 过滤出所有的NP短语（最高层的NP）
for subptree in ptree.subtrees(): # 递归遍历所有子树
	if subptree.label().find("NP")!=-1 and mytree[subptree.treeposition()[:-1]].label().find("NP")==-1:
		 print str(subptree).decode("unicode-escape")
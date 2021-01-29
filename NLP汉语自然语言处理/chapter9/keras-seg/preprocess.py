# -*- coding: utf-8 -*-
from __future__ import absolute_import
# from __future__ import print_function
import os, sys
import numpy as np 
from numpy import *
from sequencelib import *


reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

input_file = 'msr.utf8.txt'
input_text = load_file(input_file) # 读入全部文本
txtwv = [line.split() for line in input_text.split('\n') if line != '']  # 为词向量准备的文本格式
txtnltk = [w for w in input_text.split()]   # 为计算词频准备的文本格式
freqdf = freq_func(txtnltk) # 计算词频表
maxfeatures = freqdf.shape[0] # 词汇个数
#  建立两个映射字典
word2idx = dict((c, i) for c, i in zip(freqdf.word, freqdf.idx))
idx2word = dict((i, c) for c, i in zip(freqdf.word, freqdf.idx))
# word2vec
w2v = trainW2V(txtwv)
# 存向量
init_weight_wv = save_w2v(w2v,idx2word)

# 定义'U'为未登陆新字, 'P'为两头padding用途，并增加两个相应的向量表示
char_num = len(init_weight_wv)
idx2word[char_num] = u'U'
word2idx[u'U'] = char_num
idx2word[char_num+1] = u'P'
word2idx[u'P'] = char_num+1

init_weight_wv.append(np.random.randn(100,))
init_weight_wv.append(np.zeros(100,))

# 读取数据，将格式进行转换为带四种标签 S B M E
output_file = 'msr.tagging.utf8'
character_tagging(input_file, output_file)
# 分离word 和 label
with open(output_file) as f:
	lines = f.readlines()
	train_line = [[w[0] for w in line.decode('utf-8').split()] for line in lines]
	train_label = [w[2] for line in lines for w in line.decode('utf-8').split()]
# 文档转数字list
print sent2num(train_line[0],word2idx =word2idx)
# 将所有训练文本转成数字list
train_word_num = [ sent2num(line,word2idx) for line in train_line ]

print len(train_word_num)
print len(train_label)

dump(train_word_num, open('train_word_num.pickle', 'wb'))
dump(train_label, open('train_label.pickle', 'wb'))


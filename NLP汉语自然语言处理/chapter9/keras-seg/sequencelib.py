# -*- coding: utf-8 -*-
import os, sys
import numpy as np 
from numpy import *
import sys
import os
import nltk 
import codecs
import pandas as pd
from os import path
from nltk.probability import FreqDist 
from gensim.models import word2vec
from  cPickle import load, dump

reload(sys) # 设置 UTF-8输出环境
sys.setdefaultencoding('utf-8')

# 根据微软语料库计算词向量

# 读单个文本
def load_file(input_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    input_text = input_data.read()
    return input_text

# 读取目录下文本成为一个字符串
def load_dir(input_dir):
    files = list_dir(input_dir)
    seg_files_path = [path.join(input_dir, f) for f in files]
    output = []
    for txt in seg_files_path:
        output.append(load_file(txt))
    return '\n'.join(output)

# nltk  输入文本，输出词频表
def freq_func(input_txt):
    corpus = nltk.Text(input_txt) 
    fdist = FreqDist(corpus) 
    w = fdist.keys() 
    v = fdist.values() 
    freqdf = pd.DataFrame({'word':w,'freq':v}) 
    freqdf.sort('freq',ascending =False, inplace=True)
    freqdf['idx'] = np.arange(len(v))
    return freqdf

# word2vec建模
def trainW2V(corpus, epochs=20, num_features = 100,sg=1,\
             min_word_count = 1, num_workers = 4,\
             context = 4, sample = 1e-5, negative = 5):
    w2v = word2vec.Word2Vec(workers = num_workers,sample = sample,
                          size = num_features, min_count=min_word_count,
                          window = context)
    np.random.shuffle(corpus)
    w2v.build_vocab(corpus)  
    for epoch in range(epochs):
        print('epoch' + str(epoch))
        np.random.shuffle(corpus)
        w2v.train(corpus)
        w2v.alpha *= 0.9  
        w2v.min_alpha = w2v.alpha  
    print("word2vec DONE.")
    return w2v

def save_w2v(w2v, idx2word):
    # 保存词向量lookup矩阵，按idx位置存放
    init_weight_wv = []
    for i in range(len(idx2word)):
        init_weight_wv.append(w2v[idx2word[i]])
    return init_weight_wv

def character_tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/S ")
            else:
                output_data.write(word[0] + "/B ")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word)-1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()

def sent2num(sentence, word2idx = {}, context = 7):
    predict_word_num = []
    for w in sentence:
        # 文本中的字如果在词典中则转为数字，如果不在则设置为'U
        if w in word2idx:
            predict_word_num.append(word2idx[w])
        else:
            predict_word_num.append(word2idx[u'U'])
    # 首尾padding
    num = len(predict_word_num)
    pad = int((context-1)*0.5)
    for i in range(pad):
        predict_word_num.insert(0,word2idx[u'P'] )
        predict_word_num.append(word2idx[u'P'] )
    train_x = []
    for i in range(num):
        train_x.append(predict_word_num[i:i+context])
    return train_x


# -*- coding: utf-8 -*-
from __future__ import absolute_import
# from __future__ import print_function
import os, sys
import numpy as np 
from numpy import *
from sequencelib import *
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential,Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Reshape, Flatten ,Dropout
from keras.regularizers import l1,l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D,MaxPooling1D
from sklearn.cross_validation import train_test_split

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

# train_word_num = load(open('train_word_num.pickle','rb'))

nb_classes = len(np.unique(train_label))

# 建立两个字典
label_dict = dict(zip(np.unique(train_label), range(4)))
num_dict = {n:l  for l,n  in label_dict.iteritems()}
print label_dict
print num_dict
# {u'M': 2, u'S': 3, u'B': 0, u'E': 1}
# {0: u'B', 1: u'E', 2: u'M', 3: u'S'}

# 将目标变量转为数字
train_label = [ label_dict[y] for y in train_label ]
print len(train_label)

# 切分数据集
train_word_num = np.array(train_word_num)
print len(train_word_num)

train_X, test_X, train_y, test_y = train_test_split(train_word_num,train_label, train_size=0.9, random_state=1)

Y_train = np_utils.to_categorical(train_y, nb_classes)
Y_test = np_utils.to_categorical(test_y, nb_classes)

print len(train_X), 'train sequences'
print len(test_X), 'test sequences'
# 3645422 train sequences
# 405047 test sequences

# 初始字向量格式准备
init_weight = [np.array(init_weight_wv)]

batch_size = 128

maxfeatures = init_weight[0].shape[0] # 词典大小
word_dim = 100
maxlen = 7
hidden_units = 100

# stacking LSTM

print('stacking  LSTM...')
model = Sequential()
model.add(Embedding(maxfeatures, word_dim,input_length=maxlen))
model.add(LSTM(output_dim=hidden_units, return_sequences =True))
model.add(LSTM(output_dim=hidden_units, return_sequences =False))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train_X, test_X, Y_train, Y_test
print "Train..."
result = model.fit(train_X, Y_train, batch_size=batch_size, nb_epoch=20, validation_data = (test_X,Y_test), show_accuracy=True)

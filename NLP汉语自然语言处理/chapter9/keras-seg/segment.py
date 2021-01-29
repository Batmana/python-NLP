# -*- coding: utf-8 -*-
from __future__ import absolute_import
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

train_word_num = load(open('train_word_num.pickle','rb'))
train_label = load(open('train_label.pickle','rb'))
nb_classes = len(np.unique(train_label))
# 建立两个字典
label_dict = dict(zip(np.unique(train_label), range(4)))
num_dict = {n:l  for l,n  in label_dict.iteritems()}
print label_dict
print num_dict
# {u'M': 2, u'S': 3, u'B': 0, u'E': 1}
# {0: u'B', 1: u'E', 2: u'M', 3: u'S'}


# 将目标变量转为数字
train_label2 = [ label_dict[y] for y in train_label ]
train_label = train_label2 
print shape(train_label)

# 切分数据集
train_word_num = np.array(train_word_num)
print shape(train_word_num)

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

print 'stacking  LSTM...'
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


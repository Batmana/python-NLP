"""
Importing the required packages
"""
import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf

from matplotlib import pylab

from six.moves import range
from six.moves.urllib.request import urlretrieve

"""
Make sure the dataset link is copied correctly
"""
dataset_link = 'http://mattmahoney.net/dc/'
zip_file = 'text8.zip'


def data_download(zip_file):
    """
    Downloading the required file
    :param zip_file:
    :return:
    """
    if not os.path.exists(zip_file):
        zip_file, _ = urlretrieve(dataset_link + zip_file, zip_file)
        print("File downloaded successfully!")
    return None

# data_download(zip_file)

"""
提取文件夹的数据
"""
extracted_folder = 'dataset'

if not os.path.isdir(extracted_folder):
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(extracted_folder)

with open('dataset/text8') as ft_:
    full_text = ft_.read()


def text_processing(ft8_text):
    """
    替换标点符号
    :param ft8_text:
    :return:
    """
    ft8_text = ft8_text.lower()
    ft8_text = ft8_text.replace('.', '<period> ')
    ft8_text = ft8_text.replace(',', '<comma> ')
    ft8_text = ft8_text.replace('"', '<quotation> ')
    ft8_text = ft8_text.replace(';', '<semicolon> ')
    ft8_text = ft8_text.replace('!', '<exclamation> ')
    ft8_text = ft8_text.replace('?', '<question> ')
    ft8_text = ft8_text.replace('(', '<paren_l> ')
    ft8_text = ft8_text.replace(')', '<paren_r> ')
    ft8_text = ft8_text.replace('--', '<hyphen> ')
    ft8_text = ft8_text.replace(':', '<colon> ')
    ft8_text_tokens = ft8_text.split()
    return ft8_text_tokens

ft_tokens = text_processing(ft8_text=full_text)
"""
Shortlisting words with frequency more than 7
"""
word_cnt = collections.Counter(ft_tokens)
shortlisted_words = [w for w in ft_tokens if word_cnt[w] > 7]
print(shortlisted_words[:15])

print("Total number of shortlisted words:", len(shortlisted_words))
print("Unique number of shortlisted words", len(set(shortlisted_words)))


def dict_creqtion(shortlisted_words):
    """
    The function creates a dictionary of the wordds present in dataset along with their frequency order
    创建了一个单词的数据集及词频
    :param shortlisted_words:
    :return:
    """
    counts = collections.Counter(shortlisted_words)
    vocabukary = sorted(counts, key=counts.get, reverse=True)

    rev_dictionary_ = {ii: word for ii, word in enumerate(vocabukary)}
    dictionary_ = {word: ii for ii, word in rev_dictionary_.items()}

    return dictionary_, rev_dictionary_
dictionary_, rev_dictionary_ = dict_creqtion(shortlisted_words)
words_cnt = [dictionary_[word] for word in shortlisted_words]

# Skip-Gram代码
"""
Creating the threshold and performing the subsampling
定义阈值，执行二次采样
"""
thresh = 0.00005
word_counts = collections.Counter(words_cnt)
total_cnt = len(words_cnt)
freqs = {word: count / total_cnt for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(thresh / freqs[word]) for word in word_counts}

# 二次采样，p_drop代表采样率，p_drop[word] < random.random() 表示有多大几率保留该词
train_words = [word for word in words_cnt if p_drop[word] < random.random()]
print(len(train_words))

def skipG_target_set_generation(batch_, batch_index, word_window):
    """
    以所需格式创建Skip-gram输入
    The function combines the words of given word_window size next to the index, for the SkipGram model
    :param batch_:
    :param batch_index:
    :param word_window: 窗口大小
    :return:
    """
    # 返回一个随机数或随机数数组
    random_num = np.random.randint(1, word_window + 1)
    words_start = batch_index - random_num if (batch_index - random_num) > 0 else 0
    words_stop = batch_index + random_num
    window_target = set(batch_[words_start: batch_index] + batch_[batch_index + 1: words_stop + 1])

    return list(window_target)


def skipG_batch_creation(short_words, batch_length, word_window):
    """
    The function internally makes use of the skipG_target_set_generation() function and combines each of the label words in the
    shortlisted_words with the words of word_window size around
    :param short_words:
    :param batch_length:
    :param word_window:
    :return:
    """
    # // 表示整数除法
    batch_cnt = len(short_words) // batch_length
    short_words = short_words[:batch_cnt * batch_length]

    for word_index in range(0, len(short_words), batch_length):
        input_words, label_words = [], []
        word_batch = short_words[word_index: word_index + batch_length]

        # 遍历每个batch中的每个中词
        for index_ in range(len(word_batch)):
            batch_input = word_batch[index_]
            batch_label = skipG_target_set_generation(word_batch, index_, word_window)

            label_words.extend(batch_label)
            input_words.extend([batch_input] * len(batch_label))
            yield input_words, label_words

tf_graph = tf.Graph()
with tf_graph.as_default():
    input_ = tf.compat.v1.placeholder(tf.int32, [None], name='input_')
    label_ = tf.compat.v1.placeholder(tf.int32, [None, None], name='label_')

# 嵌入矩阵
with tf_graph.as_default():
    word_embed = tf.Variable(
        tf.random.uniform((len(rev_dictionary_), 300), -1, 1)
    )
    embedding = tf.nn.embedding_lookup(word_embed, input_)

"""
The code includes the following：
初始化softmax层的权重和偏置矩阵
计算负采样的损失函数
使用Adam优化器
100个词的负采样，将包含在损失函数中
设置300维度的权重
"""
vocabulary_size = len(rev_dictionary_)
with tf_graph.as_default():
    sf_weights = tf.Variable(
        tf.compat.v1.truncated_normal((vocabulary_size, 300), stddev=0.1)
    )
    sf_bias = tf.Variable(tf.zeros(vocabulary_size))
    # softmax分母计算速度慢的问题
    # 一种是Softmax-based Approaches，保持softmax层不变，但修改其架构以提高其效率，比如分层softmax；
    # 一种是Sampling-based Approaches，比如通过采样优化损失函数，来接近softmax。
    loss_fn = tf.nn.sampled_softmax_loss(
        weights=sf_weights,
        biases=sf_bias,
        labels=label_,
        inputs=embedding,
        num_sampled=100,
        num_classes=vocabulary_size
    )
    # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    cost_fn = tf.reduce_mean(loss_fn)
    optim = tf.compat.v1.train.AdamOptimizer().minimize(cost_fn)

"""
通过从期望大小的字典中随机选择16个单词来执行验证
从1000随机选择8个
使用cos距离计算两个单词的相似性
"""
with tf_graph.as_default():
    # 验证集大小
    validation_cnt = 16
    validation_dict = 100
    validation_words = np.array(
        random.sample(range(validation_cnt),
                      validation_cnt // 2)
    )

    validation_words = np.append(validation_words,
                                 random.sample(range(1000, 1000+validation_dict),  validation_cnt // 2)
                                 )

    validation_data = tf.constant(validation_words, dtype=tf.int32)
    # 归一化 embed
    normalization_embedd = word_embed / (tf.sqrt(
        tf.reduce_sum(tf.square(word_embed), 1, keepdims=True)
    ))

    validation_embed = tf.nn.embedding_lookup(normalization_embedd, validation_data)

    word_similarity = tf.matmul(validation_embed, tf.transpose(normalization_embedd))

"""
Creating the model checkpoint directory 
"""
epochs = 2
batch_length = 1000
word_window = 2

with tf_graph.as_default():
    saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session(graph=tf_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.compat.v1.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = skipG_batch_creation(train_words, batch_length, word_window)
        start = time.time()
        for x, y in batches:
            train_loss, _ = sess.run(
                [cost_fn, optim],
                feed_dict={input_: x, label_: np.array(y)[:, None]}
            )
            loss += train_loss
            if iteration % 100 == 0:
                end = time.time()
                print(
                      "Epoch {}/{}".format(e, epochs),
                      ", Iteration:{}".format(iteration),
                      ", Avg.Training loss:{:.4f}".format(loss / 100),
                      ", Processing:{:.4f} sec/batch".format((end - start) / 100)
                      )
                loss = 0
                start = time.time()

            if iteration % 2000 == 0:
                similarity_ = word_similarity.eval()
                for i in range(validation_cnt):
                    validated_words = rev_dictionary_[validation_words[i]]
                    # 最近邻居的数量
                    top_k = 8
                    nearest = (-similarity_[i, :]).argsort()[1: top_k + 1]
                    log = 'Nearest to %s' % validated_words
                    for k in range(top_k):
                        close_word = rev_dictionary_[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            iteration += 1 # 每遍历一个batch，iteration值加1
    save_path = saver.save(sess, "model_chekpoint/skipGram_text8.ckpt")
    embed_mat = sess.run(normalization_embedd)

with tf_graph.as_default():
    saver = tf.train.Saver()

with tf.compat.v1.Session(graph=tf_graph) as sess:
    """Restoring the trained netword"""
    saver.restore(sess,
                  tf.train.latest_checkpoint('model_checkpoint'))
    embed_mat = sess.run(word_embed)

from sklearn.manifold import TSNE
word_graph=250
tsne = TSNE()
word_embedding_tsne = tsne.fit_transform(embed_mat[:word_graph, :])

from keras.datasets import imdb

# load_data函数使用8个参数来自定义评论数据集
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000, index_from=3)

vocab_to_int = imdb.get_word_index()
vocab_to_int = {k: (v + 3) for k, v in vocab_to_int.items()}

vocab_to_int['<PAD>'] = 0
vocab_to_int['<GO>'] = 1
vocab_to_int['<UNK>'] = 2
int_to_vocab = {value: key for key, value in vocab_to_int.items()}
print(' '.join(int_to_vocab[id] for id in X_train[0]))


import tensorflow as tf
print(tf.__version__)

from tensorflow.python.ops import rnn, rnn_cell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl", num_words=None, skip_top=0,
                                                      maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3)

print(X_train[:2])


t = [item for sublist in X_train for item in sublist]
vocabulary = len(set(t)) + 1
# 数量分布
a = [len(x) for x in X_train]
plt.plot(a)
plt.show()

max_length = 200

x_filter = []
y_filter = []
# 不过max_length长度补0，超过则截断
for i in range(len(X_train)):
    if len(X_train[i]) < max_length:
        a = len(X_train[i])
        X_train[i] = X_train[i] + [0] * (max_length - a)
    elif len(X_train[i]) >= max_length:
        X_train[i] = X_train[i][0: max_length]
    else:
        continue
    x_filter.append(X_train[i])
    y_filter.append(y_train[i])

# 超参数定义
embedding_size = 100
n_hidden = 200
learning_rate = 0.06
training_iters = 100000
batch_size = 32
beta = 0.0001
# 定义模型体系结构和数据集相关的其他参数
# 时间步长参数
n_steps = max_length
# 分类的类数
n_classes = 2
# 自注意力MLP的隐藏层中的单元数
da = 30
# 嵌入矩阵中的行数
r = 30
display_step = 10
hidden_units = 3000

# 将训练数据集和标签转换为阵列后变换及编码的所需格式
y_train = np.asarray(pd.get_dummies(y_filter))
X_train = np.asarray([np.asarray(g) for g in x_filter])

logs_path = 'recent_logs'


class DataIterator:
    """
    产生batch数据
    """
    def __init__(self, data1, data2, batch_size):
        """
         Takes data sources and batch_size as arguments
        :param data1:
        :param data2:
        :param batch_size:
        """
        self.data1 = data1
        self.data2 = data2
        self.batch_size = batch_size
        self.iter = self.make_random_iter()

    def next_batch(self):
        try:
            idxs = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = next(self.iter)

        X = [self.data1[i] for i in idxs]
        Y = [self.data2[i] for i in idxs]

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def make_random_iter(self):
        print(self.data1)
        splits = np.arange(self.batch_size, len(self.data1), self.batch_size)
        it = np.split(np.random.permutation(range(len(self.data1))), splits)[:-1]
        return iter(it)

# 初始化权重和偏差
with tf.name_scope("weights"):
    Win = tf.Variable(
        tf.compat.v1.random_uniform([n_hidden * r, hidden_units], -1/np.sqrt(n_hidden), 1/np.sqrt(n_hidden)),
        name="W-input"
    )

    Wout = tf.Variable(
        # tf随机函数
        tf.random_uniform([hidden_units, n_classes], -1/np.sqrt(n_hidden), 1/np.sqrt(n_hidden)),
        name="W-out"
    )

    Ws1 = tf.Variable(
        tf.random_uniform([da, n_hidden], -1/np.sqrt(da), 1/np.sqrt(da)),
        name="Ws1"
    )

    Ws2 = tf.Variable(
        tf.random_uniform([r, da], -1/np.sqrt(r), 1/np.sqrt(r)),
        name="Ws2"
    )


with tf.name_scope("biases"):
    biasesout = tf.Variable(tf.random_normal([n_classes]), name='biases-out')
    biasesin = tf.Variable(tf.random_normal([hidden_units]), name='biases-in')


with tf.name_scope("input"):
    x = tf.placeholder("int32", [32, max_length], name='x-input')
    y = tf.placeholder("int32", [32, 2], name="y-input")

with tf.name_scope("embedding"):
    embedings = tf.Variable(
        tf.random_uniform([vocabulary, embedding_size], -1, 1),
        name='embeddings'
    )
    embed = tf.nn.embedding_lookup(embedings, x)


def length(sequence):
    """

    :param sequence: 句子
    :return:
    """
    # Computing maximum of elements across dimensions of a tensor
    used = tf.sign(
        tf.reduce_max(tf.abs(sequence), reduction_indices=2)
    )
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    
    return length


# 下面的代码重算权重和偏差
with tf.variable_scope("forward", reuse=True):
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden)

with tf.name_scope("model"):
    outputs, states = rnn.dynamic_rnn(lstm_fw_cell,
                                      embed,
                                      sequence_length=length(embed),
                                      dtype=tf.float32,
                                      time_major=False)
    # 下一步我们将隐藏向量与Ws1 相乘，重塑Ws1
    h = tf.nn.tanh(
        tf.transpose(tf.reshape(tf.matmul(Ws1, tf.reshape(outputs, [n_hidden, batch_size * n_steps])), [da, batch_size, n_steps]), [1, 0, 2])
    )

    a = tf.reshape(
        tf.matmul(Ws2, tf.reshape(h, [da, batch_size * n_steps])),
        [batch_size, r, n_steps]
    )

    def fn3(a, x):
        """

        :param a:
        :param x:
        :return:
        """
        print(a)
        print(x)

        return tf.nn.softmax(x)

    h3 = tf.scan(fn3, a)

with tf.name_scope("flattening"):
    h4 = tf.matmul(h3, outputs)
    last = tf.reshape(h4, [-1, r * n_hidden])

with tf.name_scope("MLP"):
    tf.nn.dropout(last, .5, noise_shape=None, seed=None, name=None)
    pred1 = tf.nn.sigmoid(tf.matmul(last, Win) + biasesin)
    pred = tf.matmul(pred1, Wout) + biasesout

# 定义损失函数及优化器
with tf.name_scope("cross"):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + beta * tf.nn.l2_loss(Ws2)
    )

with tf.name_scope("Train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # 计算梯度
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gvs]
    optimizer.apply_gradients(capped_gvs)
    optimized = optimizer.minimize(cost)

with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(
        tf.argmax(pred, 1),
        tf.argmax(y, 1)
    )

    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32)
    )

tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "summary operation" which we can execute in a session
summary_op = tf.summary.merge_all()
training_iter = DataIterator(X_train, y_train, batch_size)
init = tf.global_variables_initializer()

# This could give warning if in case the required port is being used already
# Running the command again or releasing the port before the subsequent run should solve the purpose
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = training_iter.next_batch()
        sess.run(optimized,
                 feed_dict={x: batch_x, y: batch_y})
        # Executing the summary operation in the session
        summary = sess.run(
            summary_op, feed_dict={x: batch_x, y: batch_y}
        )

        writer.add_summary(summary, step * batch_size)
        if step % display_step == 2:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.2f}".format(acc * 100) + "%")
        step += 1
    print("Optimization Finished!")


    



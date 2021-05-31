"""
https://www.jianshu.com/p/d42283e0e2d0
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time

print(tf.__version__)

# vocabulary.txt
# ID 词
lines = open("InsuranceQnA/vocabulary.txt",
             encoding='utf-8',
             errors='ignore').read().split('\n')
# 读取问题
# InsuranceQAquestionanslabelraw.encode
# 主题    问题词ID列表  答案ID
conv_lines = open("InsuranceQnA/InsuranceQAquestionanslabelraw.encoded",
                  encoding='utf-8',
                  errors='ignore').read().split('\n')

# 读取答案
# 答案ID    词ID列表
conv_lines1 = open("InsuranceQnA/InsuranceQAlabel2answerraw.encoded",
                   encoding='utf-8',
                   errors='ignore').read().split('\n')

# The print command shows the token value associated with each of the words in the 3 datasets
print("------------Vocabulary-----------------")
print(lines[:2])

print("------------Question-----------------")
print(conv_lines[:2])

print("------------Answer-----------------")
print(conv_lines1[:2])

vocab_lines = lines
question_lines = conv_lines
answer_lines = conv_lines1

# 依据分配给问题和答案的ID，将问题与其对应的答案组合起来
id2line = {}
for line in vocab_lines:
    _line = line.split('\t')
    if len(_line) == 2:
        id2line[_line[0]] = _line[1]

# 为问题和答案分词，同时做答案和问题的映射
convs, ansid = [], []
for line in question_lines[:-1]:
    _line = line.split('\t')
    ansid.append(_line[2].split(' '))
    convs.append(_line[1])

convs1 = []
for line in answer_lines[:-1]:
    _line = line.split('\t')
    convs1.append(_line[1])

print(convs[:2])
print(ansid[:2])
print(convs1[:2])
questions, answers = [], []
for a in range(len(ansid)):
    for b in range(len(ansid[a])):
        questions.append(convs[a])

for a in range(len(ansid)):
    for b in range(len(ansid[a])):
        answers.append(convs1[int(ansid[a][b]) - 1])

ques, ans = [], []
m = 0
while m < len(questions):
    i = 0
    a = []
    while i < len(questions[m].split(' ')):
        a.append(id2line[questions[m].split(' ')[i]])
        i = i + 1
    ques.append(' '.join(a))
    m = m + 1

n = 0
while n < len(answers):
    j = 0
    b = []
    while j < len(answers[n].split(' ')):
        b.append(id2line[answers[n].split(' ')[j]])
        j = j + 1
    ans.append(' '.join(b))
    n = n + 1

limit = 0
for i in range(limit, limit + 5):
    print(ques[i])
    print(ans[i])
    print("----")

print(len(questions))
print(len(answers))


def clean_text(text):
    """

    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

clean_questions = []
for question in ques:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in ans:
    clean_answers.append(clean_text(answer))

limit = 0
for i in range(limit, limit + 5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()

# Remove questions and answers that are shorter than 1 words and longer than 100 words
# 删除少于1个单词且超过100个单词的问题和答案
min_line_length, max_line_length = 2, 100

# Filter out the questions that are too short/long
short_questions_temp, short_answers_temp = [], []
i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length \
            and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions, short_answers = [], []
i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length \
        and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1
print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used:{}".format(round(len(short_questions) / len(questions), 4) * 100))


def pad_sentence_batch(sentence_batch, vocab_to_int):
    """
    填充句子
    :param sentence_batch:
    :param vocab_to_int:
    :return:
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

# 对新形成的训练数据集的词汇表中的单词进行映射，并为每个单词分配频率标记
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
# 删除频率较低单词
# 临界值
threshold = 1
count = 0
for k, v in vocab.items():
    if v >= threshold:
        count += 1

print("词表大小:", len(vocab))
print("我们将使用的词表大小", count)

# 创建词典
questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1

answers_vocab_to_int = {}
word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1

# 添加四种特殊标记到词表
# GO 标记开始
# EOS 标记结束
# UNK 未知标记，这一标记用于替换词汇表中频率太低的单词
# PAD 填充标记
codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']
for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1
for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))

# 减少有效词汇表大小，方法是简单地将它限制在一个小数字范围内，并用UNK标记替换词汇表外的单词
# 数据输入模型前，必须将句子的每个单词转换成唯一的整数。
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)
answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)

# 进一步检查被替换UNK标记的单词数量
word_count = 0
unk_count = 0
for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int['<UNK>']:
            unk_count += 1
        word_count += 1

for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int['<UNK>']:
            unk_count += 1
        word_count += 1
unk_ratio = round(unk_count / word_count, 4) * 100
print("单词总数", word_count)
print("UNK数目", unk_count)
print("UNK占比", unk_ratio)

# 根据问题中的单词数量，排序问题和答案
# 这种方式将有助于稍后使用的填充方法，同时可以加快训练，减小损失函数
sorted_questions = []
sorted_questions1 = []
sorted_answers = []
sorted_answers1 = []

for length in range(1, max_line_length + 1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_questions1.append(short_questions[i[0]])
            sorted_answers.append(answers_int[i[0]])
            sorted_answers1.append(short_answers[i[0]])

print(len(sorted_questions))
print(len(sorted_questions1))
print(len(sorted_answers))
print(len(sorted_answers1))
print()

for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print(sorted_questions1[i])
    print(sorted_answers1[i])
    print()

# 随机检查一对
print(sorted_questions[1547])
print(sorted_questions1[1547])
print(sorted_answers[1547])
print(sorted_answers1[1547])

# 设置模型参数
epochs = 50
batch_size = 64
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probaility = 0.75

tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

from model import SeqModel

seq_model = SeqModel()
input_data, targets, lr, keep_prob = seq_model.model_inputs()
# 句子长度是每一个batch的最大长度
sequence_length = tf.compat.v1.placeholder_with_default(max_line_length, None, name='sequence_length')

# 将矩阵的维度输出为一维
input_shape = tf.shape(input_data)

train_logits, inference_logits = seq_model.seq2seq_model(tf.reverse(input_data, [-1]),
                                                         targets,
                                                         keep_prob,
                                                         batch_size,
                                                         sequence_length,
                                                         len(answers_vocab_to_int),
                                                         len(questions_vocab_to_int),
                                                         encoding_embedding_size,
                                                         decoding_embedding_size,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_vocab_to_int)

tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # 计算损失函数
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length])
    )

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # 进行剪枝，处理梯度消失
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients if grad is not None]

    train_op = optimizer.apply_gradients(capped_gradients)


def batch_data(questions, answers, batch_size):
    """
    为问题和答案创建批次
    :param questions:
    :param answers:
    :param batch_size:
    :return:
    """
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i: start_i + batch_size]
        answers_batch = answers[start_i: start_i + batch_size]
        pad_questions_batch = np.array(pad_questions_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_answers_batch(answers_batch, answers_vocab_to_int))

        yield pad_questions_batch, pad_answers_batch

#  数据15% 用于验证，剩下85% 用于训练模型
train_valid_split = int(len(sorted_questions) * 0.15)
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_questions[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))

# 设置训练参数并初始化声明的变量
display_step = 20
stop_early = 0
# 如果连续5次后验证损失减少，请停止训练
stop = 5
validation_check = ((len(train_questions)) // batch_size // 2) - 1
total_train_loss = 0
summary_valid_loss = []
checkpoint = "./bert_model.ckpt"
sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(batch_data(train_questions, train_answers, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch, targets:answers_batch, lr: learning_rate,
             sequence_length: answers_batch.shape[1], keep_prob: keep_probaility}
        )
        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print("Epoch {:>3}/{} Batch {:>4}/{} - Loss:{:>6.3f}. Seconds:{:>4.2f}".format(
                epoch_i, epochs, batch_i,
                len(train_questions) // batch_size, total_train_loss / display_step,
                batch_time * display_step
            ))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                    cost,
                    {input_data: questions_batch,
                     targets: answers_batch,
                     lr: learning_rate,
                     sequence_length: answers_batch.shape[1],
                     keep_prob:1}
                )
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
            print("Valid losss:{>6.3f}, Seconds: {>5.2f}".format(avg_valid_loss, batch_time))

            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            summary_valid_loss.append(avg_valid_loss)

            if avg_valid_loss <= min(summary_valid_loss):
                print("New Record!")
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)
            else:
                print(" No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break
                if stop_early == stop:
                    print("Stopping Training.")
                    break
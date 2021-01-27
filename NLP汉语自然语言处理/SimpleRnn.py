# -*- coding=UTF-8 -*
"""
SimpleRnn核心代码，将书中代码重写，加强记忆
"""
import numpy as np
import copy

class SimpleRnn(object):

    def __init__(self):
        """
        构造函数
        """
        pass

    def center(self, dataset, largest_number):
        """
        核心代码
        :return:
        """
        # 隐藏层向量维度
        hidden_dim = 100
        # 输入层向量维度
        binary_dim = 200
        # 主程序，训练过程
        for i in xrange(maxiter):
            # 在实际应用中，可以从训练数据中查询到一个样本：生成形如[a]->[b]->[c]的样本
            a, a_int, b, b_int, c, c_int = gensample(dataset, largest_number)

            # 初始化一个空的二进制数据，用来存储神经网络的预测值
            d = np.zeros_like(c)

            # 重置全局误差
            overallError = 0

            # 记录layer 2的导数值
            layer_2_deltas = list()
            layer1_values = list()
            # 初始化时无值，存储一个全零的向量
            layer1_values.append(np.zeros(hidden_dim))

            # 正向传播过程，逐个bit位(0,1)的遍历二进制数字
            for position in xrange(binary_dim):
                # 数组索引 7,6,5,...0
                indx = binary_dim - position - 1
                # X是样本集的记录，来自a[i]b[i]; y是样本集的标签，来自c[i]
                X = np.array([[a[indx], b[indx]]])
                Y = np.array([[c[indx]]]).T

                # 隐含层(input ~+ prev_hidden)
                # 从输入层出传播到隐含层：输入层数据 *（输入层到隐含层的权值）
                # 1. np.dot(X, synapse_I);
                # 从上一次的隐含层[-1] 到大当前的隐含层；上一次的隐含层权值 * 当前隐含层的权值
                # 2. np.dot(layer1_values[-1], synapse_h)
                # 3. sigmoid(input + prev_hidden)
                layer_1 = sigmoid(np.dot(X, synapse_T) + np.dot(layer1_values[-1], synapse_h))

                # 输出层
                # 它从隐含层传播到输出层，即输出一个预测值
                # np.dot(layer_1, synapse_O)
                layer_2 = sigmoid(np.dot(layer_1, synapse_O))

                # 计算预测误差
                layer_2_error = y - layer_2
                # 保留输出层每个时刻的误差，用于反向传播
                layer_2_deltas.append((layer_2_error) * dlogit(layer_2))
                # 计算二进制位对的误差绝对值的总和，标量
                overallError += np.abs(layer_2_error[0])
                # 存储预测的结果 - 显示使用
                d[indx] = np.round(layer_2[0][0])
                # 存储隐含层对的权值，以便在下次时间迭代中能使用
                layer1_values.append(copy.deepcopy(layer_1))

            # 初始化下一隐含层的误差
            future_layer_1_delta = np.zeros(hidden_dim)
            # 反向传播，从最后一个时间点开始，反向一直到第一个：position 索引 0,1,2,....7
            for position in xrange(binary_dim):
                X = np.array([[a[position]], b[position]])
                # 从列表中取出当前到隐含层。从最后一层开始，-1，-2，-3
                layer_1 = layer1_values[-position - 1]
                # 从代表中取出当前层的前一隐含层
                prev_layer_1 =layer1_values[-position - 2]
                # 取出当前输出层的误差
                layer_2_delta = layer_2_deltas[-position - 1]
                # 计算当前隐含层的误差
                # 下一隐含层误差 * 隐含层权重
                # funture_layer_1_delta.dot(synapse_h.T)
                # 当前输出层误差 * 输出层权重
                # layer_2_deltas.dot(synapse_O.T)
                # 当前隐含层的权重
                # dlogit(layer_1)
                layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_O.T) * dlogit(layer_1))

                # 反向更新权重: 更新顺序输出层-> 隐含层 -> 输入层
                # 输入层reshape为2d的数组
                # np.atleast_2d:
                synapse_O_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                synapse_I_update += X.T.dot(layer_1_delta)

                # 下一隐含层的误差
                future_layer_1_delta = layer_1_delta

            # 更新三个权值
            synapse_I += synapse_I_update * alpha
            synapse_O += synapse_O_update * alpha
            synapse_h += synapse_h_update * alpha

            # 所有权值更新项归零
            synapse_O_update *= 0
            synapse_h_update *= 0
            synapse_I_update *= 0

            # 逐次打印输出
            showresult(j, overallError, d, c, a_int, b_int)
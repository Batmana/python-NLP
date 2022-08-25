"""
Tf的示例代码
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hello = tf.constant("Hello,Tensors!")
sess = tf.compat.v1.Session()
print(sess.run(hello))

# Mathematical computation
# 数学运算
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))

import numpy as np
# 创建numpy数组
mat_1 = 10 * np.random.random_sample((3, 4))
mat_2 = 10 * np.random.random_sample((4, 6))

# 创建一对常量运算，并包括上述制成的矩阵
tf_mat_1 = tf.constant(mat_1)
tf_mat_2 = tf.constant(mat_2)

# Multiplying Tensorflow matrices with matrix multiplication
# 将Tensorflow矩阵与矩阵相乘
tf_mat_prod = tf.matmul(tf_mat_1, tf_mat_2)
sess = tf.compat.v1.Session()

# run（）执行必需的操作，并执行将输出存储在'mult_matrix'变量中的请求
mult_matrix = sess.run(tf_mat_prod)
print(mult_matrix)

# Performing constant operations with the addition and subtraction of two constant
# 用两个常数的加法和减法执行常数运算
a = tf.constant(10)
b = tf.constant(20)
print("Addition of constant 10 and 20 is %i" % sess.run(a + b))
print("Subtration of constant 10 and 20 is %i" % sess.run(a - b))
sess.close()

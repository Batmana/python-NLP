"""
Keras 的示例代码
"""

# 从keras导入所需的库，层和模型
import keras
from keras.layers import Dense
# Keras Sequential 顺序模型
# 顺序模型是多个网络层的线性堆叠
from keras.models import Sequential
import numpy as np

# Dataset link: # http://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Centers
# Save the dataset as a.cssv file
train_ = np.genfromtxt('transfusion.csv', delimiter=',')
print(train_.shape)
X = train_[:, 0:4]
Y = train_[:, 4]
print(X)

# 使用Keras创建我们的第一个MLP模型
# mlp_keras = Sequential()
# mlp_keras.add(Dense(8,
#                     input_dim=4,
#                     init='uniform',
#                     activation='relu'
#                     ))
# mlp_keras.add(Dense(6,
#                     init="uniform",
#                     activation='relu'
#                     ))
# mlp_keras.add(Dense(1,
#                     init="uniform",
#                     activation='sigmoid'
#                     ))
#
# mlp_keras.compile(loss="binary_crossentropy",
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
# mlp_keras.fit(X, Y, nb_epoch=200, batch_size=8, verbose=0)
# accurary = mlp_keras.evaluate(X, Y)
# print("Accurary: %.2f%%" % (accurary[1] * 100))

# 使用Keras创建我们的第一个MLP模型
from keras.optimizers import SGD
opt = SGD(lr=0.01)

mlp_keras = Sequential()
mlp_keras.add(Dense(8,
                    input_dim=4,
                    kernel_initializer='uniform',
                    bias_initializer='uniform',
                    activation='relu'
                    ))
mlp_keras.add(Dense(6,
                    kernel_initializer="uniform",
                    bias_initializer='uniform',
                    activation='relu'
                    ))
mlp_keras.add(Dense(1,
                    kernel_initializer="uniform",
                    bias_initializer='uniform',
                    activation='sigmoid'
                    ))

mlp_keras.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])

mlp_keras.fit(X, Y,
              validation_split=0.3,
              epochs=150,
              batch_size=10,
              verbose=0)
results_optim = mlp_keras.evaluate(X, Y)
print("Accurary: %.2f%%" % (results_optim[1] * 100))




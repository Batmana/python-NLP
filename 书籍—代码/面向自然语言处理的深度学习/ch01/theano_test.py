"""
Theano 是一个开源项目，暂时已经停止维护
"""
import theano
import theano.tensor as T
import numpy
from theano import function
# 定义矩阵
x = T.dscalar('x')
y = T.dscalar('y')

# 'x' 和 'y' are instance of TensorVariable， and are of dscalar theano type
print(type(x))
print(x.type)
print(T.dscalar)

# 'z' represents the sum of 'x' and 'y' variables.
# Theano's pp function, pretty-print out, is used to display the computation of the variable 'z'
z = x + y
from theano import pp
print(pp(z))
# 'f' is a numpy.ndarray of zero dimensions,
# which takes input as the first argument, and output as the second argument
# 'f' is being compiled in C code
f = function([x, y], z)
print(f(6, 10))
print(numpy.allclose(f(10.3, 5.4), 15.7))



import numpy as np
a = np.array([1, 4, 5, 8], float)
print(type(a))
print(a)
a[0] = 5
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]], float)
print(b[0, 1])
print(b.shape)
print(b.dtype)
print(len(b))

print(2 in b)
print(0 in b)

c = np.array(range(12), float)
print(c)
print(c.shape)
print('---')
c = c.reshape((2, 6))
print(c)
print(c.shape)

# 矩阵填充
c.fill(0)
print(c)

# 转置
print(c.transpose())
# 展平
print(c.flatten())
m = np.array([1, 2], float)
n = np.array([3, 4, 5, 6], float)
# 矩阵连接
p = np.concatenate((m, n))
print(p)
print(p.shape)

q = np.array([1, 2, 3], float)
# newaxis 增加新维度
print(q[:, np.newaxis].shape)

# Numpy 还有其他函数，如zeros, zeros_like, ones_like, identity, eye
# Numpy中的乘法是元素间到相乘，而不是矩阵乘法
a1 = np.array([[1, 2], [3, 4], [5, 6]], float)
a2 = np.array([-1, 3], float)
print(a1 + a2)

# Numpy 提供了一些可直接用于数组的函数
# sum(元素的和)、prod(元素的积)
# mean(元素的平均值)、var(元素的方差)
# std(元素的标准差)、argmin（数组中最小元素的索引）
# argmax(数组中最大元素的索引)、sort(对元素排序)、unique(数组中的唯一元素)
a3 = np.array([[0, 2], [3, -1], [3, 5]], float)
print(a3.mean(axis=0))
print(a3.mean(axis=1))

# numpy 提供了用于测试数组中是否存在某些数值的函数
# nonzero 检查非零元素
# isnan 检查 非数字 元素
# isfinite 检查有限元素
a4 = np.array([1, 3, 0], float)
print(np.where(a != 0, 1/a, a))

# 生成不同长度的随机数，使用numpy中的random函数
print(np.random.rand(2, 3))



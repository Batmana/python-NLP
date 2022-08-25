import numpy as np
import scipy

# scipy中的线性代数子包
from scipy import linalg
# Matrix Creation
mat_ = np.array([[2, 3, 4], [4, 9, 10], [10, 5, 6]])
print(mat_)
# Deteminant of the matrix
# 计算矩阵的行列式
print(linalg.det(mat_))
# Inverse of the matrix
# 计算矩阵的逆
inv_mat = linalg.inv(mat_)
print(inv_mat)

# 用于执行奇异值分解并存储各个组成部分
# Singular Value Decompostion
comp_1, comp_2, comp_3 = linalg.svd(mat_)
print(comp_1)
print(comp_2)
print(comp_3)

# Scipy.stats 是一个大型子包，包含各种各样到的统计分布处理函数，可用于操作不同类型的数据集
# Scipy Stats module
from scipy import stats
# Generating a random sample of size 20 from normal
# distribution with mean 3 and standard deviation 5
# 生成均值3，标准值5的 20个随机样本
rvs_20 = stats.norm.rvs(3, 5, size=20)
print('--------')
print(rvs_20, '\n-------')

# Computing the CDF of Beta distribution with a=100 and b=130 as shape parameters at random variable 0.41
# 计算Beta分布的CDF
cdf_ = scipy.stats.beta.cdf(0.41, a=100, b=130)

print(cdf_)


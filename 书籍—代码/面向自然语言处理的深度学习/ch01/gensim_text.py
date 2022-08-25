"""
gensim 主要用于主题建模和文档相似性处理
"""
import gensim
from gensim.models import Word2Vec
min_count = 0
size = 50
window = 2
sentents = "bitcoin is an innovative payment network and a new kind of money"
sentents = sentents.split()
print(sentents)

model = Word2Vec(sentents, min_count=min_count, size=size, window=window)
print(model)
# 单词a的向量
print(model['a'])

# 可以从https://github.com/mmihaltz/word2vec-GoogleNews-vectors下载模型
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
sentence = ["I", "hope", "it", "is", "going", "good", "for", "you"]
vectors = [model[w] for w in sentence]
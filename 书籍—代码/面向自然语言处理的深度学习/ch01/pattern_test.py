"""
pattern 适用于各种NLP任务
例如词类标注器、n-gram搜索、情感分析、WordNet和机器学习（例如向量空间建模、k均值聚类、朴素贝叶斯、KNN、SVM分类器）
py37 报错：解决：https://github.com/clips/pattern/issues/243
"""
from pattern.text.en import tag
tweet = "I hope it is going good for you!"
tweet_1 = tweet.lower()
tweet_tags = tag(tweet_1)
print(tweet_tags)
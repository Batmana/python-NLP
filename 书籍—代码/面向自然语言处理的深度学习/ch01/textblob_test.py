"""
TextBlob是一个用于处理文本数据的Python库。
可应用于深入常见的NLP任务，例如词类标注、名词短语提取、情感分析、分类等。
"""
from textblob import TextBlob

# Taking a statement  as input
statement = TextBlob("My home is far away from my school.")
# Calculating the sentiment attached with the statement
# 计算情绪 polarity 定义句子中的消极性或积极性   subjectivity 暗示句子的表达是含糊的 还是肯定的
print(statement.sentiment)


# 可以利用TextBlob进行标注
text = '''How about you and I go together on a walk far away
from this place, discussing the things we have never discussed
on Deep Learning and Natural Language Processing'''
blob_ = TextBlob(text)
print(blob_)

print(blob_.tags)

# 利用TextBlob 更正拼写错误
sample_ = TextBlob("I thinkk the model needs to be trained more!")
print(sample_.correct())

# 该包提供了翻译模块
# Language Translation
# 使用了谷歌API翻译
lang_ = TextBlob("Voulez-vous apprendre le francais?")
print(lang_.translate(from_lang='fr', to='en'))

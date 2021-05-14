"""
NLTK在处理语料库、分类文本、分析语言结构等。
"""
import nltk

# Tokenization
sent_ = "I am almost dead this time"
tokens_ = nltk.word_tokenize(sent_)
print(tokens_)

# 确保安装了wordnet.
# import nltk
# nltk.download('wordnet)
# Synonyms 同义词
from nltk.corpus import wordnet
word_ = wordnet.synsets("spectacular")
print(word_)
# Printing the meaning along of each of synonyms
# 打印同义词的语义
print(word_[0].definition())
print(word_[1].definition())
print(word_[2].definition())

# 它可以执行词干提取和词性还原
# Stemming 词干提取
# 波特词干算法分词器
from nltk.stem import PorterStemmer
# Create the stemmer object
stemmer = PorterStemmer()
print(stemmer.stem('decreases'))

# Lemmatization 词形还原
# WordNetLemmatizer 词形归并
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('decreases'))

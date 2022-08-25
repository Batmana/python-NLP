
# 1.4.1 使用正则表达式进行文本搜索
import re
words = ['very', 'nice', 'lecture', 'day', 'moon']
expression = '|'.join(words)
print(expression)
print(re.findall(expression, 'i attended a very nice lecture last year', re.M))

# 1.4.2 将文本转换为列表
text_file = "data.txt"
with open(text_file) as f:
    words = f.read().split()
print(words)
#  whole text as single element of the list
# 全文作为列表的单个元素
f = open(text_file, 'r')
words_ = f.readlines()
print(words_)

# 1.4.3 文本预处理
sentence = 'John has been selected for the trial phase this time.Congrats!'
sentence = sentence.lower()
print(sentence)
# defining the positive and negative words explicitly
# 明确定义正面和负面词
positive_words = ['awesome', 'good', 'nice', 'super', 'fun', 'delightful', 'congrats']
negative_words = ['awful', 'lame', 'horrible', 'bad']
sentence = sentence.replace('!', '')
print(sentence)

words = sentence.split(' ')
print(words)
result = set(words) - set(positive_words)
print(result)

# 1.4.4 从网页中获取文本
import urllib3
from bs4 import BeautifulSoup
pool_project = urllib3.PoolManager()
target_url = "https://www.cnblogs.com/feifeifeisir/p/10627474.html"
response_ = pool_project.request('GET', target_url)
final_html_txt = BeautifulSoup(response_.data)
print(final_html_txt)

# 1.4.5 移除停止词
import nltk
from nltk import word_tokenize
sentence = "This book is about Deep Learning and Natural Language Processing!"
tokens = word_tokenize(sentence)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

new_token = [w for w in tokens if not w in stop_words]
print(new_token)

# 1.4.6 计数向量化
# 计数向量化是Sklearn的工具，可以接收任何大量的文本，将每个特殊的单词作为特征返回，并计算每个单词在文本中出现的次数
from sklearn.feature_extraction.text import CountVectorizer
texts = ["Ramiess sings classic songs",
         "he listens to old pop",
         "and rock music",
         "and also listens to classical songs"]
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())

# 1.4.7 TF-IDF分数
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
X = vect.fit_transform(texts)
print(X.todense())

# 1.4.8 文本分类器
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

data = [
    ("I love my country", 'pos'),
    ("This is an amazing place!", 'pos'),
    ("I do not like the smell of this place", 'neg'),
    ("I do not like this restaurant", 'neg'),
    ("I am tiredd of hearing your nonsense", 'neg'),
    ("I always aspire to be like him", "pos"),
    ("It's a horrible performance.", "neg")
]

model = NaiveBayesClassifier(data)
print(model.classify("It's an awesome place"))
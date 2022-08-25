"""
Spacy提供了非常快速和准确的句法分析功能。还提供了命名实体识别和可以随时访问词向量的功能。
它使用Cython语言编写，包含各种训练模型，包括语言词汇表、语法、词向量转换和实体识别。
"""
import spacy
nlp = spacy.load('en_core_web_sm')
william_wikidef = """william was the son of King william 
∥ and Anna Pavlova of Russia. On the abdication of his
grandfather William I in 1840,he became the Prince of Orange.
On the death of his father in 1849, he succeeded as king of the 
Netherlands.William married his cousin Sophie of Wurttemberg
in 1839 and they had three sons,William,Maurice, and 
Alexander,all of whom predeceased him"""

nlp_william = nlp(william_wikidef)
print([(i, i.label_, i.label) for i in nlp_william.ents])

# Scapy 提供依赖性解析，可以进一步从文本中提取名词短语
# Noun Phrase extraction
# 名词短语提取
senten_ = nlp("The book deals with NLP")
for noun_ in senten_.noun_chunks:
    print(noun_)
    print(noun_.text)
    print('-----')
    print(noun_.root.dep_)
    print('------')
    print(noun_.root.head.text)
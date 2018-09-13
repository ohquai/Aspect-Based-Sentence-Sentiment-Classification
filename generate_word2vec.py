# -*- coding: utf-8 -*-

import pandas as pd
from gensim.models import Word2Vec, word2vec
import logging
from gensim.models.word2vec import LineSentence

df = pd.read_csv('D:/Data/sentence/train.csv', encoding='utf8')
sentences = set(df['sentence_seq'])
line_sent = []
with open('D:/Data/sentence/train.txt', 'w', encoding='utf8') as f:
    for s in sentences:
        # sentence = s.split(' ')
        # line_sent.append(s.split(' '))  #句子组成list
        f.write(s+'\n')
f.close()
# print(len(line_sent))
# print(line_sent[0])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('D:/Data/sentence/train.txt')  # 加载语料

model = Word2Vec(sentences, size=300, window=5, min_count=2, sg=1)
model.save('D:/Data/sentence/word2vec.model')
# model = word2vec.Word2Vec.load("text8.model")

# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format('D:/Data/sentence/word2vec.txt', binary=False)
# model = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

# model = Word2Vec.load('word2vec_model')
y2 = model.most_similar(u"动力", topn=20)
for item in y2:
    print(item[0], item[1])
print("--------\n")

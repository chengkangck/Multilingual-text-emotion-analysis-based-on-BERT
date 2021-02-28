from pyltp import Segmentor
from pyltp import Postagger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB

import numpy as np

import configparser

import re
import time
import os
import sys

config = configparser.ConfigParser()
config.read('../config.ini')

# nltk.download()
# load stopwords
def load_stopwords(swlist):
    with open(os.path.join(config['path']['INPUT_PATH'], 'stopwords_zh.txt'), 'r') as sw_dic:
        sw_content = sw_dic.read()
    for sw in sw_content.splitlines():
        swlist.append(sw)


# load news.txt
def load_corpus(expected_tags, segmentor, postager, swlist, corpus, labels):
    if not os.path.exists('../output/segwords.txt'):
        for fname in ['sentiment.train', 'sentiment.dev', 'test.data']:
            with open('../input/' + fname, "r") as cf:
                for line in cf.readlines():
                    if (len(line.strip()) != 0):
                        items = line.strip().split('\t')
                        words = list(segmentor.segment("".join(items[:-1]) if fname[0] != 't' else "".join(items[:])))
                        if fname[0] != 't' and len(words) == 0:
                            continue 
                        tags = list(postager.postag(words))
                        if fname[0] != 't':  
                            labels.append(int(items[-1]))
                        seglist = []
                        for i in range(len(words)):
                            if tags[i] in expected_tags and words[i] not in swlist:
                                seglist.append(words[i])
                        corpus.append(" ".join(seglist))
        if not os.path.exists('../output'):
            os.makedirs('../output')
        with open('../output/segwords.txt', 'w') as sf:
            for doc in corpus:
                sf.write(doc + '\n')
    else:
        with open('../output/segwords.txt', 'r') as sf:
            for i, doc in enumerate(sf):
                corpus.append(doc.strip())
        for fname in ['sentiment.train', 'sentiment.dev']:
            with open('../input/' + fname, "r") as cf:
                for line in cf.readlines():
                    if len(line1.strip()) != 0:
                        items = line.strip().split('\t')
                        labels.append(int(items[-1]))

if __name__ == "__main__":
    # 过滤的时候保留的词性
    expected_tags = ["n", "nh", "ni", "nl", "ns", "nt", "nz", \
                            "a", "e", "i", "v", "z", "d"]
    # 停词表
    swlist = []
    # 语料库
    corpus = []
    # 标签
    labels = []

    starttime = time.time()
    print('开始时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    timestamp = time.time()
    print("加载停词表...")
    load_stopwords(swlist)
    print("加载停词表耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("分词...")
    cws_model_path = os.path.join(config['path']['LTP'], 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化分词器实例
    segmentor.load(cws_model_path)
    pos_model_path = os.path.join(config['path']['LTP'], 'pos.model')  # 词性标模型注模型，模型名称为'pos.model'
    postagger = Postagger()  # 初始化词性标注实例
    postagger.load(pos_model_path)  # 加载模型
    load_corpus(expected_tags, segmentor, postagger, swlist, corpus, labels)
    print("分词耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("词频统计...")
    tf_vectorizer = TfidfVectorizer(min_df=0.005, max_df=0.2, max_features= 2000)
    model = tf_vectorizer.fit(corpus)
    print("词频统计耗时：", time.time() - timestamp, "s")

    timestamp = time.time()
    print("NB训练...")
    clf = GaussianNB()
    clf.fit(model.transform(corpus[:-200000]).toarray(), labels)
    print("NB训练耗时：", time.time() - timestamp, "s")
    
    timestamp = time.time()
    print("NB预测...")
    res = clf.predict(model.transform(corpus[-200000:]).toarray())
    print("NB预测耗时：", time.time() - timestamp, "s")

    with open('../output/baseline.csv', 'w') as bf:
        bf.write('ID,Expected\n')
        for id, label in enumerate(res):
            bf.write(str(id) + ',' + str(label) + '\n')
    print("结束时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

import pandas as pd
import numpy as np

import jieba
from gensim.models import Word2Vec



class Preprocess():
    def __init__(self, sentences, args, tokens=None):
        jieba.load_userdict(args.jieba_lib) # 載入繁體辭典
        if tokens:
            for token in tokens:
                jieba.add_word(token)
        self.embed_dim = args.word_dim # 詞向量維度
        self.seq_len = args.seq_len # 句子長度
        self.wndw_size = args.wndw # 給前後幾個詞來預測中間的詞
        self.word_cnt = args.cnt # 只對在訓練資料中出現超過一定次數的詞做詞轉向量
        self.save_name = 'word2vec' # 詞向量模型名稱
        self.index2word = [] # list 按照詞的編號排序
        self.word2index = {} # dict key為詞 value為詞的編號
        self.vectors = [] # 詞向量
        self.unk = "<UNK>" # 不認識的詞
        self.pad = "<PAD>" # 填補句子長度用
        self.data = self.tokenize(sentences) # 切詞後的句子

    '''結巴切詞'''
    def tokenize(self, sentence):
        print("=== Jieba cutting")
        tokens = []
        for idx, s in enumerate(sentence):
            tmp = []
            for item in jieba.cut(s):
                tmp.append(item)
            tokens.append(tmp)
        
        return tokens

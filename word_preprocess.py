import pandas as pd
import numpy as np

import jieba
from gensim.models import Word2Vec



class Preprocess():
    def __init__(self, sentences, args, tokens=None):
        jieba.load_userdict(args.jieba_lib) # 載入繁體辭典
        if tokens: # 增加自定義詞彙
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


    ''' 詞轉向量 ''' 
    # e.g. self.word2index['你'] = 1 
    # e.g. self.index2word[1] = '你'
    # e.g. self.vectors[1] = '你'的向量
    def get_embedding(self, load=False):
            print("=== Get embedding")
            # Get Word2vec word embedding
            if load:
                embed = Word2Vec.load(self.save_name)
            else:
                embed = Word2Vec(self.data, size=self.embed_dim, 
                                window=self.wndw_size, min_count=self.word_cnt, 
                                iter=16, workers=8)
                embed.save(self.save_name)
            # Add special tokens    
            self.add_embedding(self.pad)
            self.add_embedding(self.unk)
            # Index tokens
            for i, word in enumerate(embed.wv.vocab):
                print('=== get words #{}'.format(i+1), end='\r')              
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
                self.vectors.append(embed.wv[word])
            print('\n')
            self.vectors = np.array(self.vectors)
            print("=== total words: {}".format(len(self.vectors)))
            return self.vectors


    def add_embedding(self, word):
        # Add random uniform vector
        vector = np.random.uniform(0, 1, (self.embed_dim, ))
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors.append(vector)

    
    ''' 對句子進行編碼並固定長度 '''
    def get_indices(self):   
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]     
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):      
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
            # if word in word2index append word index into sentence_indices
            # if word not in word2index append unk index into sentence_indices
                if word in self.index2word:
                    sentence_indices.append(self.word2index[word])
                else:
                    sentence_indices.append(self.word2index[self.unk])
            sentence_indices = self.pad_to_len(sentence_indices, 
                                                self.seq_len, 
                                                self.word2index[self.pad])
            all_indices.append(sentence_indices)
        print('\n')

        return np.array(all_indices)         
        

                
    def pad_to_len(self, arr, padded_len, padding=0):
        """ 
        if len(arr) < padded_len, pad arr to padded_len with padding.
        If len(arr) > padded_len, truncate arr to padded_len.
        Example:
            pad_to_len([1, 2, 3], 5, 0) == [1, 2, 3, 0, 0]
            pad_to_len([1, 2, 3, 4, 5, 6], 5, 0) == [1, 2, 3, 4, 5]
        Args:
            arr (list): List of int.
            padded_len (int)
            padding (int): Integer used to pad.
        Return:
            arr (list): List of int with size padded_len.
        """
        if len(arr) < padded_len:
            arr.extend([self.word2index[self.pad]] * (padded_len - len(arr)))
        elif len(arr) > padded_len:
            arr = arr[0:padded_len]

        return arr
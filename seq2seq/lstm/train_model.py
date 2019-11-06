import sys
sup_dir = 'D:\\ML\\NLP\\booking.com'
sys.path.append(sup_dir)

import os
import argparse

import numpy as np
import pandas as pd

from word_preprocess import Preprocess


def read_data(data_dir):
    comment = []; reply = []
    for path, dirs, files in os.walk(data_dir, topdown=False):
        for idx, file in enumerate(files):              
            df = pd.read_csv(os.path.join(path, file)).dropna()
            df = df[(df['comment_bad']!=' ') & (df['response']!=' ')]
            # 飯店統稱為飯店名稱
            if len(df.index)!=0:
                # print(f'open file: {dirpath} {file}')
                name = df['hotel_name'].iloc[0]
                sub = '飯店名稱'
                for idx, item in df['response'].items():
                    tmp = item.replace(name, sub)
                    df['response'].loc[idx] = tmp
                comment.extend(df['comment_bad'].tolist())
                reply.extend(df['response'].tolist())
  
    return comment, reply


def main(args):
    '''讀取負評及回覆 並將回覆的頭尾加入特殊字元'''
    comment, reply = read_data(args.data_dir)
    for idx, item in enumerate(reply):
        reply[idx] = '句子開頭' + item + '句子結尾'

    # 詞轉向量 load=True代表使用已經訓練好的詞向量模型(word2vec)
    tokens = ['句子開頭', '句子結尾', '飯店名稱']
    preprocess = Preprocess(comment+reply, args, tokens=tokens)
    print(preprocess.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 一定要輸入的變數 
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('data_dir', type=str, help='[Input] training data directory')

    # 選擇輸入的變數 
    parser.add_argument('--lr', default=0.0001, type=float) # learning rate
    parser.add_argument('--batch', default=64, type=int) # batch size
    parser.add_argument('--epoch', default=64, type=int) # epoch
    parser.add_argument('--seq_len', default=50, type=int) # 句子長度
    parser.add_argument('--word_dim', default=100, type=int) # 詞向量維度
    parser.add_argument('--hidden_dim', default=32, type=int) # lstm輸出維度
    parser.add_argument('--wndw', default=5, type=int) # 給前後幾個詞來預測中間的詞
    parser.add_argument('--cnt', default=3, type=int) # 只對在訓練資料中出現超過一定次數的詞做詞轉向量
    args = parser.parse_args()

    main(args)
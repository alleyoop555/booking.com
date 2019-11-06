import sys
sup_dir = 'D:\\ML\\NLP\\booking.com'
sys.path.append(sup_dir)

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from word_preprocess import Preprocess
from for_model import build_model, use_callback, MyGenerator



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
    # 讀取負評及回覆 並將回覆的頭尾加入特殊字元 
    comment, reply = read_data(args.data_dir)
    for idx, item in enumerate(reply):
        reply[idx] = '句子開頭' + item + '句子結尾'

    #  詞轉向量 load=True代表使用已經訓練好的詞向量模型(word2vec) 
    tokens = ['句子開頭', '句子結尾', '飯店名稱']
    preprocess = Preprocess(comment+reply, args, tokens=tokens)
    word_vector = preprocess.get_embedding(load=False)
    indices = preprocess.get_indices()
    comment_index = indices[:len(comment)]; reply_index = indices[len(comment):]
    print(f'=== Comment and Reply shape: {comment_index.shape} {reply_index.shape}')

    # 去掉句子長度超過seq_len的評論及回應 
    index_end = preprocess.word2index['句子結尾']
    index_pad = preprocess.word2index['<PAD>']
    remain_index = []
    for idx, item in enumerate(zip(comment_index, reply_index)):
        if (item[0][-1]==index_pad) & \
           (item[1][-1]==index_end or item[1][-1]==index_pad):
            remain_index.append(idx)
    comment_index = comment_index[remain_index]
    reply_index = reply_index[remain_index]
    print(f'=== Comment and Reply shape: {comment_index.shape} {reply_index.shape}')

    # 去掉重複的句子
    reply_index, indices = np.unique(reply_index, axis=0, return_index=True)
    comment_index = comment_index[indices]
    print(f'=== Comment and Reply shape: {comment_index.shape} {reply_index.shape}')

    # 
    model = build_model(args, word_vector)
    callback_list = use_callback(args)
    C_train, C_test, R_train, R_test = train_test_split(
        comment_index, reply_index, 
        test_size=0.2, 
        random_state=10
        )
    model.fit_generator(
        MyGenerator(args, C_train, R_train, word_vector, index_pad),  
        steps_per_epoch=round(C_train.shape[0]/args.batch), epochs=args.epoch, 
        validation_data=MyGenerator(args, C_test, R_test, word_vector, index_pad),
        validation_steps=round(C_test.shape[0]/args.batch), 
        callbacks=callback_list
        )



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
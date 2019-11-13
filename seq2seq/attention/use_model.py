import sys
sup_dir = 'D:\\ML\\NLP\\booking.com'
sys.path.append(sup_dir)

import argparse
import numpy as np

from tensorflow.python.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Model
from keras.utils import plot_model, print_summary

from word_preprocess import Preprocess
from custom_layer import AttentionLayer

def wrd2vec(args, comment):
    preprocess = Preprocess(comment, args)
    word_vector = preprocess.get_embedding(load=True)
    index = preprocess.get_indices()
    
    return index, preprocess

def reply_model():
    model = load_model('./model/lstm_s2s.h5', custom_objects={'AttentionLayer': AttentionLayer})
    print('Model loaded')

    """ Encoder model """
    encoder_input= model.inputs[0]   
    encoder_out, encoder_h, encoder_c = model.get_layer('encoder_lstm').output
    encoder_state = [encoder_h, encoder_c]
    encoder_model = Model([encoder_input], [encoder_out, encoder_h, encoder_c])
    plot_model(encoder_model, './image/lstm_encoder.png', show_shapes=True)
    print('encoder ok')

    """ Decoder model """
    decoder_input = model.inputs[1] 
    decoder_ini_h = Input(shape=(encoder_h.shape[1],), name='decoder_ini_h')
    decoder_ini_c = Input(shape=(encoder_c.shape[1],), name='decoder_ini_c')
    decoder_ini_state = [decoder_ini_h, decoder_ini_c]
    decoder_attn_input = Input(shape=encoder_out.shape[1:], name='decoder_attn_input')

    decoder_emb = model.get_layer('embedding').get_output_at(1)
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_out, decoder_h, decoder_c = decoder_lstm(decoder_emb, initial_state=decoder_ini_state)
    attn_layer = model.get_layer('attention_layer')
    attn_out, attn_state = attn_layer([decoder_attn_input, decoder_out])

    decoder_concat = model.get_layer('concat_layer')([decoder_out, attn_out])
    decoder_pred = model.get_layer('time_distributed_layer')(decoder_concat)
    decoder_model = Model(inputs=[decoder_input, decoder_ini_h, decoder_ini_c, decoder_attn_input],
                          outputs=[decoder_pred, decoder_h, decoder_c, attn_state])
    plot_model(decoder_model, './image/lstm_decoder.png', show_shapes=True)
    print('decoder ok')

    return encoder_model, decoder_model

def decode_sequence(args, encoder, decoder, comment_index, preprocess):

    # Encode the input as state vectors.
    # index = np.expand_dims(comment_index, axis=0)
    encoder_out, encoder_h, encoder_c = encoder.predict([comment_index])
    states_value = encoder_out
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, args.seq_len,)) 
    # Populate the first character of target sequence with the start character.
    target_seq[0][0] = preprocess.word2index['句子開頭']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''

    for i in range(args.seq_len):
        output_tokens, h, c, attn_state = decoder.predict([target_seq, encoder_h, encoder_c, encoder_out])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = preprocess.index2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '句子結尾' or len(decoded_sentence) > args.seq_len):
            return decoded_sentence

        # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, 1))
        target_seq[0][i] = sampled_token_index

    return decoded_sentence

def main(args):
    if args.comment:
        comment = [' '.join(args.comment) if args.comment else ' ']
        comment_index, preprocess = wrd2vec(args, comment)

        print('=== show comment cut')
        print(comment)   
        for idx in range(comment_index.shape[1]):
            nn = comment_index[0][idx]
            print(preprocess.index2word[nn], end=' | ')
        print('\n')
        
        print('=== show reply result')
        encoder, decoder = reply_model()
        decoded_sentence = decode_sequence(args, encoder, decoder, comment_index, preprocess)
        print(decoded_sentence)

    else:
        print('No comment No reply')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('--comment', default=None, type=str, nargs='+')

    parser.add_argument('--seq_len', default=50, type=int) # 句子長度
    parser.add_argument('--word_dim', default=100, type=int) # 詞向量維度
    parser.add_argument('--wndw', default=5, type=int) # 給前後幾個詞來預測中間的詞
    parser.add_argument('--cnt', default=3, type=int) # 只對在訓練資料中出現超過一定次數的詞做詞轉向量
    
    args = parser.parse_args()
    main(args)
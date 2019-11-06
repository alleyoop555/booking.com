import sys
sup_dir = 'D:\\ML\\NLP\\booking.com'
sys.path.append(sup_dir)

import argparse
import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.utils import print_summary, plot_model

from word_preprocess import Preprocess



def wrd2vec(args, comment):
    preprocess = Preprocess(comment, args)
    word_vector = preprocess.get_embedding(load=True)
    index = preprocess.get_indices()
    
    return index, preprocess


def reply_model():
    model = load_model('./model/lstm_s2s.h5')
    
    encoder_input= model.inputs[0]   
    encoder_output = model.get_layer('lstm_1').output
    encoder_model = Model([encoder_input], encoder_output)
    plot_model(encoder_model, './image/lstm_encoder.png', show_shapes=True)

    decoder_input = model.inputs[1]
    decoder_input_h = Input(shape=(encoder_output[1].shape[1],), name='input_3')
    decoder_input_c = Input(shape=(encoder_output[2].shape[1],), name='input_4')
    decoder_input_state = [decoder_input_h, decoder_input_c]
    decoder_embed = model.get_layer('embedding_1').get_output_at(1)
    decoder_lstm = model.get_layer('lstm_2')
    lstm_output, decoder_output_h, decoder_output_c = decoder_lstm(decoder_embed, 
                                            initial_state=decoder_input_state)
    decoder_output_state = [decoder_output_h, decoder_output_c]                                        
    decoder_output = model.get_layer('dense_1')(lstm_output)
    decoder_model = Model([decoder_input] + decoder_input_state, 
                        [decoder_output] + decoder_output_state)
    plot_model(decoder_model, './image/lstm_decoder.png', show_shapes=True)

    return encoder_model, decoder_model


def decode_sequence(args, encoder, decoder, comment_index, preprocess):
    # Encode the input as state vectors.
    # index = np.expand_dims(comment_index, axis=0)
    states_value = encoder.predict([comment_index])[1:]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1)) 
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = preprocess.word2index['句子開頭']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = preprocess.index2word[sampled_token_index]
        
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '句子結尾' or len(decoded_sentence) > args.seq_len):
            stop_condition = True
        else:
            decoded_sentence += sampled_char
            
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

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
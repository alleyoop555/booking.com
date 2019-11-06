import random

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional
from keras.layers import concatenate, Dense
from keras.initializers import Constant
from keras.utils import plot_model, print_summary
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import keras.backend as K



def build_model(args, word_vector):
    # 模型架構
    encoder_input = Input(shape=(None,))
    embedding = Embedding(
        word_vector.shape[0], args.word_dim,
        embeddings_initializer=Constant(word_vector),
        trainable=False,
        mask_zero=True
        )
    encoder_embedding = embedding(encoder_input)
    encoder = LSTM(args.hidden_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]
    encoder_model = Model([encoder_input], encoder_states)

    decoder_input = Input(shape=(None,))
    decoder_embedding = embedding(decoder_input)
    decoder = LSTM(args.hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(word_vector.shape[0], activation='softmax')(decoder_outputs)
    model = Model([encoder_input, decoder_input], decoder_dense)
    plot_model(model, './image/lstm_model.png', show_shapes=True) 

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model


def use_callback(args):
    checkpoint = ModelCheckpoint('./model/lstm_s2s.h5', 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        )
    call = EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=args.epoch/8, 
        restore_best_weights=True
        )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.8, 
        patience=args.epoch/16, 
        min_lr=args.lr*0.1
        )

    callback_list = [checkpoint, call, reduce_lr]

    return callback_list


def MyGenerator(args, comment, reply, word_vector, index_pad):
    while True:
        idx = random.choices(range(comment.shape[0]), k=args.batch)
        target = np.zeros((len(idx), args.seq_len, 1), dtype='float32')
        for i in range(len(idx)):
            for j in range(args.seq_len):
                if j==args.seq_len-1:
                    target[i, j, 0] = index_pad
                else:
                    target[i, j, 0] = reply[idx[i], j+1]

        yield [comment[idx], reply[idx]], target
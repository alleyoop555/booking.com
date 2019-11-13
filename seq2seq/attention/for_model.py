import random

import numpy as np

from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.python.keras.layers import Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, Callback
from keras.utils import plot_model

from custom_layer import AttentionLayer



def build_model(args, word_vector):
    # 模型架構
    encoder_inputs = Input(batch_shape=(None, args.seq_len), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(None, args.seq_len), name='decoder_inputs')
    
    embedding = Embedding(
                        word_vector.shape[0], args.word_dim,
                        embeddings_initializer=Constant(word_vector),
                        input_length=args.seq_len,
                        trainable=False,
                        mask_zero=True
                        )

    # Encoder 
    encoder_emb = embedding(encoder_inputs)
    encoder_lstm = LSTM(args.hidden_dim, return_sequences=True, 
                        return_state=True, name='encoder_lstm')
    encoder_out, encoder_h, encoder_c = encoder_lstm(encoder_emb)
    encoder_state = [encoder_h, encoder_c]

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_emb = embedding(decoder_inputs)
    decoder_lstm = LSTM(args.hidden_dim, return_sequences=True, return_state=True, 
                        name='decoder_lstm')
    decoder_out, decoder_h, decoder_c = decoder_lstm(decoder_emb, 
                                    initial_state=encoder_state)
  
    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')\
                                        ([decoder_out, attn_out])

    # Dense layer
    dense = Dense(word_vector.shape[0], activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    plot_model(model, './image/lstm_model.png', show_shapes=True) 
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy'
        )

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
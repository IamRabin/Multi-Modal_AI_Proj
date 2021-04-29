
import config
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding

with open(config.vocab, "r") as voc:
    vocab = json.load(voc)


def create_glove_embeddings():
    embeddings_index = {}
    f = open("/source/Data_preprocessed/new_data_embed300.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((config.MAX_NUM_WORDS, config.EMBEDDING_DIM))
    for word, i in vocab.items():
        if i >= config.MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return tf.keras.layers.Embedding(input_dim=config.MAX_NUM_WORDS, output_dim=config.EMBEDDING_DIM,
                                     input_length=config.MAX_SEQ_LENGTH,
                                     weights=[embedding_matrix],
                                     trainable=True)


def create_channel(x, filter_size, feature_map):
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = Flatten()(x)
    return x



import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam
from keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import cnn_model
import config
import utils


def run():
    with open(config.text_data, "rb") as handle:
        text = pickle.load(handle)
    with open(config.img_data, "rb")as handle:
        img = pickle.load(handle)
    with open(config.vocab, "r") as voc:
        vocab = json.load(voc)

    original_data = pd.read_csv(config.raw_text_labels)
    ids = list(set(list(text.keys())) & set(list(img.keys())))
    text = [text[patient] for patient in ids]
    img = [img[patient] for patient in ids]
    y = [original_data[original_data['ID'] == patient].Labels.item() for patient in ids]
    # Split the dataset for text
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        text, y, test_size=0.2, random_state=42)
    # Split the dataset for the img
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        img, y, test_size=0.2, random_state=42)

    histories = []

    for i in range(config.RUNS):
        print('Running iteration %i/%i' % (i+1, config.RUNS))

        emb_layer = None
        if USE_GLOVE:
            emb_layer = create_glove_embeddings()

        model_txt = build_cnn(
            embedding_layer=emb_layer,
            num_words=config.MAX_NUM_WORDS,
            embedding_dim=config.EMBEDDING_DIM,
            filter_sizes=config.FILTER_SIZES,
            feature_maps=config.FEATURE_MAPS,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dropout_rate=config.DROPOUT_RATE
        )

        model_txt.compile(
            loss='binary_crossentropy',
            optimizer=Adadelta(clipvalue=3),
            metrics=['accuracy']
        )

        history = model_txt.fit(
            np.array(X_train_text), np.array(y_train),
            epochs=NB_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=(np.array(X_test_text), np.array(y_test)),
            callbacks=[ModelCheckpoint('best_models/text_model-%i.h5' % (i+1), monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='min'),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01)
                       ]
        )
        print()
        histories.append(history.history)

    image_input = Input(shape=(224, 224, 3))
    base_model = DenseNet121(include_top=True, weights='/content/drive/MyDrive/medical_multimodal_with_transfer_learning/best_models/CheXNet_Densenet121_weights.h5',
                             input_tensor=image_input, input_shape=None, pooling=None, classes=14)
    last_layer = base_model.get_layer('avg_pool').output
    x = BatchNormalization()(last_layer)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.5)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    model_img = Model(image_input, out)
    model_img.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    hist = model_img.fit(np.array(X_train_img), np.array(y_train), batch_size=32,
                         epochs=3, verbose=1, validation_data=(np.array(X_test_img), np.array(y_test)))
    model_img.save_weights('best_models/img_model-1.h5')

    x_in = Input(shape=(config.MAX_SEQ_LENGTH,), dtype='int32')
    channels = []
    embedding_layer = create_glove_embeddings()
    emb_layer = embedding_layer(x_in)
    if config.DROPOUT_RATE:
        emb_layer = Dropout(config.DROPOUT_RATE)(emb_layer)
    for ix in range(len(config.FILTER_SIZES)):
        x = create_channel(emb_layer, config.FILTER_SIZES[ix], config.FEATURE_MAPS[ix])
        channels.append(x)
    # Concatenate all channels
    x = concatenate(channels)
    text_last_layer = concatenate(channels)
    ###########################################

    # Image Sub-model
    image_input = Input(shape=(224, 224, 3))
    base_model = DenseNet121(include_top=True, weights='/content/drive/MyDrive/medical_multimodal_with_transfer_learning/best_models/CheXNet_Densenet121_weights.h5',
                             input_tensor=image_input, input_shape=None, pooling=None, classes=14)
    last_layer = base_model.get_layer('avg_pool').output
    img_last_layer = BatchNormalization()(last_layer)
    ###########################################

    # Fusion
    fusion = concatenate([text_last_layer, img_last_layer])
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    multi_model = Model([x_in, image_input], out)
    ###########################################
    multi_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    hist = multi_model.fit(x=([np.array(X_train_text), np.array(X_train_img)]), y=np.array(
        y_train), batch_size=32, epochs=3, verbose=1, validation_data=(([np.array(X_test_text), np.array(X_test_img)]), np.array(y_test)))

    multi_model.save_weights('best_models/multi_model-1.h5')


if __name__ == "__main__ ":
    run()

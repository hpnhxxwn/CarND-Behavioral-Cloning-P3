import csv
import tensorflow as tf
from keras.layers import Conv2D, Dropout, Dense, Flatten, Lambda, Activation, MaxPooling2D
import keras
import numpy as np
from keras.models import Model, load_model
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from utils import INPUT_SHAPE, image_batch_generator
import argparse
import os



def nvidia_model():
    '''
    based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x / 255., input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    # model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train(model, args):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    if os.path.isfile('model.h5'):
        print('Loading model')
        model = load_model('model.h5')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    train_gen = image_batch_generator(256)
    val_gen = image_batch_generator()

    history = model.fit_generator(train_gen,
                              args.samples_per_epoch,
                              # samples_per_epoch=20000,
                              nb_epoch=args.nb_epoch,
                              nb_val_samples=args.nb_epoch,
                              validation_data=val_gen,
                              verbose=1,
                              callbacks=[checkpoint])

    model.save('model.h5')

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    # parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    # parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    # parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=bool,   default=True)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # data = load_data(args)
    model = nvidia_model()
    train(model, args)


if __name__ == '__main__':
    main()

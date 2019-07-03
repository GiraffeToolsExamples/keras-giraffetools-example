'''
Created by the GiraffeTools Keras generator.
Warning, here be dragons.

'''

import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# Model
def NeuralNet(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  ):

    dense_1 = Dense(
      10,
      activation='softmax',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense_1'
    )

    dropout_1 = Dropout(
      0.5,
      name='dropout_1'
    )(dense_1)

    dense = Dense(
      128,
      activation='relu',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense'
    )(dropout_1)

    flatten = Flatten(
      name='flatten'
    )(dense)

    dropout = Dropout(
      0.25,
      name='dropout'
    )(flatten)

    max_pooling2_d = MaxPooling2D(
      pool_size=(2, 2),
      padding='valid',
      name='max_pooling2_d'
    )(dropout)

    conv2_d_1 = Conv2D(
      32,
      (3,3),
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='conv2_d_1'
    )(max_pooling2_d)

    conv2_d = Conv2D(
      32,
      (3,3),
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='conv2_d'
    )(conv2_d_1)


    # Creating model
    _model = tf.keras.models.Model(
      inputs  = [conv2_d],
      outputs = [dense_1]
    )

    _model.compile(
      optimizer = optimizer,
      loss      = loss,
      metrics   = metrics
    )

    # Returning model
    return _model

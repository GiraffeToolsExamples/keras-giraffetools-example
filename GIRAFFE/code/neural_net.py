'''
Created by the GiraffeTools Keras generator.
Warning, here be dragons.

'''

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Model
def NeuralNet(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  ):

    dense = Dense(
      10,
      activation='softmax',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense'
    )

    dropout = Dropout(
      0.5,
      name='dropout'
    )(dense)

    dense_2 = Dense(
      128,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense_2'
    )(dropout)

    flatten_1 = Flatten(
      name='flatten_1'
    )(dense_2)

    dropout_2 = Dropout(
      0.25,
      name='dropout_2'
    )(flatten_1)

    max_pooling2_d_1 = MaxPooling2D(
      pool_size=(2, 2),
      padding='valid',
      name='max_pooling2_d_1'
    )(dropout_2)

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
    )(max_pooling2_d_1)

    conv2_d_2 = Conv2D(
      32,
      (3,3),
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='conv2_d_2'
    )(conv2_d)


    # Creating model
    _model = tf.keras.models.Model(
      inputs  = [conv2_d_2],
      outputs = [dense]
    )

    _model.compile(
      optimizer = optimizer,
      loss      = loss,
      metrics   = metrics
    )

    # Returning model
    return _model

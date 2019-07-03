'''
Created by the GiraffeTools Tensorflow generator.
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

    conv2d = Conv2D(
      32,
      (3,3),
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='conv2d'
    )

    conv2d_1 = Conv2D(
      32,
      (3,3),
      strides=(1, 1),
      padding='valid',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='conv2d_1'
    )(conv2d)

    max_pooling2d = MaxPooling2D(
      pool_size=(2, 2),
      padding='valid',
      name='max_pooling2d'
    )(conv2d_1)

    dropout_1 = Dropout(
      0.25,
      name='dropout_1'
    )(max_pooling2d)

    flatten = Flatten(
      name='flatten'
    )(dropout_1)

    dense_1 = Dense(
      128,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense_1'
    )(flatten)

    dropout_3 = Dropout(
      0.5,
      name='dropout_3'
    )(dense_1)

    dense_2 = Dense(
      10,
      activation='softmax',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      name='dense_2'
    )(dropout_3)


    # Creating model
    _model = tf.keras.models.Model(
      inputs  = [conv2d],
      outputs = [dense_2]
    )

    _model.compile(
      optimizer = optimizer,
      loss      = loss,
      metrics   = metrics
    )

    # Returning model
    return _model

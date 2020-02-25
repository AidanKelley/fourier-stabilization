import tensorflow as tf
from tensorflow import keras

from FixedWeight import FixedWeight

from BiasLayer import BiasLayer

def custom_sigmoid(tensor):
  return 2*tf.math.sigmoid(tensor) - 1


# the first category is 0 the second is 1
def get_model(activation=custom_sigmoid):
  model = keras.Sequential([
    keras.layers.Dense(16, activation=activation, input_shape=(135,)),
    keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# this is the model with no softmax at the end of it
def get_logit_model(activation=custom_sigmoid):
  model = keras.Sequential([
    keras.layers.Dense(16, activation=activation, input_shape=(135,)),
    keras.layers.Dense(2) # no activation function
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def get_fixed_weight_model(activation=custom_sigmoid):
  model = keras.Sequential([
      keras.layers.Dense(16, activation="linear", use_bias=False, input_shape=(135,), trainable=False),
      BiasLayer(activation=activation),
      keras.layers.Dense(2, activation="softmax", trainable=False)
    ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

@tf.custom_gradient
def sign(tensor):

  def grad(dx):
    return dx

  return tf.math.sign(tensor), grad





















  

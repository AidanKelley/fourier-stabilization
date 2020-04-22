import tensorflow as tf
from tensorflow import keras

from FixedWeight import FixedWeight

from BiasLayer import BiasLayer

import numpy as np

optimizer='adam'

def custom_sigmoid(tensor):
  return 2*tf.math.sigmoid(tensor) - 1


# the first category is 0 the second is 1
def get_model(input_shape, output_shape, activation, layer_size):
  model = keras.Sequential([
    keras.layers.Dense(layer_size, activation=activation, input_shape=input_shape),
    keras.layers.Dense(output_shape, activation='softmax')
  ])

  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# this is the model with no softmax at the end of it
def get_logit_model(input_shape, output_shape, activation, layer_size):
  model = keras.Sequential([
    keras.layers.Dense(layer_size, activation=activation, input_shape=input_shape),
    keras.layers.Dense(output_shape) # no activation function
  ])

  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def get_fixed_weight_model(input_shape, output_shape, activation, layer_size):
  # cute trick for only freezing the weights
  # define a new layer without biases and make activation linear
  # then a layer after it which is just biases and has the activation function
  # this then behaves exactly like a Dense layer  
  model = keras.Sequential([
    keras.layers.Dense(layer_size, activation="linear", use_bias=False, input_shape=input_shape, trainable=False),
    BiasLayer(activation=activation),
    keras.layers.Dense(output_shape, activation="softmax", trainable=False)
  ])

  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def get_new_mnist_model(x_train, y_train, activation_name, layer_size, flavor=None):
  output_shape = np.amax(y_train) + 1
  print(output_shape)
  return get_new_model_helper(x_train.shape[1:], output_shape, activation_name, layer_size, flavor)

def get_new_model(x_train, activation_name, flavor=None):
  return get_new_model_helper(x_train.shape[1:], 2, activation_name, 16, flavor)

def get_new_model_helper(input_shape, output_shape, activation_name, layer_size, flavor=None):
  if activation_name == "custom_sigmoid":
    activation = custom_sigmoid
  elif activation_name == "sign":
    activation = sign
  else:
    activation = keras.activations.deserialize(activation_name)

  if flavor is None or flavor == "model":
    model = get_model(input_shape, output_shape, activation, layer_size)
  elif flavor == "logit_model":
    model = get_logit_model(input_shape, output_shape, activation, layer_size)
  elif flavor == "fixed_weight_model":
    model = get_fixed_weight_model(input_shape, output_shape, activation, layer_size)
  else:
    raise ValueError(f"invalid value {flavor} for argument flavor")

  return model


def load_mnist_model(x_train, y_train, file_name, layer_size, flavor=None):
  return load_model_helper(x_train, y_train, file_name, layer_size, flavor)

def load_model(x_train, file_name, flavor=None):
  return load_model_helper(x_train, None, file_name, 16, flavor)

def load_model_helper(x_train, y_train, file_name, layer_size, flavor=None):
  colon_index = file_name.find(":")
  assert(colon_index > 0)

  weights_file = file_name[0:colon_index]
  activation_name = file_name[colon_index+1:]




  if y_train is None:
    model = get_new_model(x_train, activation_name, flavor=flavor)
  else:
    model = get_new_mnist_model(x_train, y_train, activation_name, layer_size, flavor)

  model.build(x_train.shape)
  model.load_weights(weights_file)

  return model, activation_name



@tf.custom_gradient
def sign(tensor):

  def grad(dx):
    return dx

  return tf.math.sign(tensor), grad





















  

import tensorflow as tf
from tensorflow import keras

from FixedWeight import FixedWeight

from BiasLayer import BiasLayer

def custom_sigmoid(tensor):
  return 2*tf.math.sigmoid(tensor) - 1


# the first category is 0 the second is 1
def get_model(input_shape=(135,), activation=custom_sigmoid):
  model = keras.Sequential([
    keras.layers.Dense(16, activation=activation, input_shape=input_shape),
    keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# this is the model with no softmax at the end of it
def get_logit_model(input_shape=(135,), activation=custom_sigmoid):
  model = keras.Sequential([
    keras.layers.Dense(16, activation=activation, input_shape=input_shape),
    keras.layers.Dense(2) # no activation function
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def get_fixed_weight_model(input_shape=(135,), activation=custom_sigmoid):
  model = keras.Sequential([
      keras.layers.Dense(16, activation="linear", use_bias=False, input_shape=input_shape, trainable=False),
      BiasLayer(activation=activation),
      keras.layers.Dense(2, activation="softmax", trainable=False)
    ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def get_new_model(x_train, activation_name, flavor=None):
  if activation_name == "custom_sigmoid":
    activation = custom_sigmoid
  elif activation_name == "sign":
    activation = sign
  else:
    activation = keras.activations.deserialize(activation_name)

  if flavor is None or flavor == "model":
    model = get_model(x_train.shape[1:], activation)
  elif flavor == "logit_model":
    model = get_logit_model(x_train.shape[1:], activation)
  elif flavor == "fixed_weight_model":
    model = get_fixed_weight_model(x_train.shape[1:], activation)
  else:
    raise ValueError(f"invalid value {flavor} for argument flavor")

  return model


def load_model(x_train, file_name, flavor=None):
  colon_index = file_name.find(":")

  if colon_index >= 0:
    weights_file = file_name[0:colon_index]
    activation_name = file_name[colon_index+1:]
  else:
    weights_file = file_name
    activation_name = None

  model = get_new_model(x_train, activation_name, flavor=flavor)
  model.build(x_train.shape)
  model.load_weights(weights_file)

  return model, activation_name



@tf.custom_gradient
def sign(tensor):

  def grad(dx):
    return dx

  return tf.math.sign(tensor), grad





















  

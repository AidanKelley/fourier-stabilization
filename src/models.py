import tensorflow as tf
from tensorflow import keras

from .FixedWeight import FixedWeight

from .BiasLayer import BiasLayer

import numpy as np

optimizer='adam'

# defines the scaled sigmoid / custom sigmoid activation function
def custom_sigmoid(tensor):
  """
  The scaled sigmoid / custom sigmoid activation function, which is a standard sigmoid
  scaled and translated so that the limit is +-1 as we approach +- infinity

  ...

  Parameters
  ----------
  tensor
    the input to the activation function

  Returns
  -------
  a tensor of the same shape
  """
  return 2*tf.math.sigmoid(tensor) - 1


# generic function to get the model 
def get_model(input_shape, output_shape, activation, layer_size):
  """
  A function to get the "standard" model, which has a softmax final layer

  ...


  Parameters
  ----------
  input_shape: Tuple
    a tuple of the shape of the input (if there are d features, this is (d,))
  output_shape: Tuple
    a tuple of the output shape (if there are x classes, this is (x,))
  activation
    the activation function to use
  layer_size: int
    the number of hidden layers

  Returns
  -------
  a keras Sequential model
  """
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
  """
    Replicates the functionality of "get_model", except the last layer is logits instead of soft_max
  """
  model = keras.Sequential([
    keras.layers.Dense(layer_size, activation=activation, input_shape=input_shape),
    keras.layers.Dense(output_shape) # no activation function
  ])

  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def get_fixed_weight_model(input_shape, output_shape, activation, layer_size):
  """ 
    Replicates the functionality of "get_model", except all weights except for biases in the hidden layer are frozen (cannot be trained)
  """

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

def get_frozen_layer_model(input_shape, output_shape, activation, layer_size):
  """ 
    Replicates the functionality of "get_model", except the weights in the first layer (but not the biases) are frozen (meaning they cannot be trained)
  """

  # cute trick for only freezing the weights
  # define a new layer without biases and make activation linear
  # then a layer after it which is just biases and has the activation function
  # this then behaves exactly like a Dense layer  
  model = keras.Sequential([
    keras.layers.Dense(layer_size, activation="linear", use_bias=False, input_shape=input_shape, trainable=False),
    BiasLayer(activation=activation),
    keras.layers.Dense(output_shape, activation="softmax", trainable=True)
  ])

  model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def get_new_mnist_model(x_train, y_train, activation_name, layer_size, flavor=None):
  """
  Gets a model that will classify the MNIST dataset

  ...


  Parameters
  ----------
  x_train: Numpy Array
    This is the MNIST training X set (used for dimensions)
  y_train: Numpy Array
    This is the MNIST training Y set (used for dimensions and to find number of classes)
  activation_name: string
    the name of the activation function you wish to use (see get_new_model_helper)
  layer_size: int
    the number of neurons in the hidden layer
  flavor: "model" (or None) | "logit_model" | "fixed_weight_model" | "frozen_layer_model"
    defines which of the three flavors of models to use (see get_new_model_helper)

  Returns
  -------
  A keras Sequential model
  """

  # get the number of output classes
  output_shape = np.amax(y_train) + 1
  print(output_shape)
  return get_new_model_helper(x_train.shape[1:], output_shape, activation_name, layer_size, flavor)

def get_new_model(x_train, activation_name, flavor=None):
  """
  Gets a new model to be used with Hidost or PDFRateB
  
  ...

  Parameters
  ----------
  x_train: Numpy Array
    The training input data to be used for this model
  activation_name: string
    the name of the activation function you wish to use (see get_new_model_helper)
  flavor: "model" (or None) | "logit_model" | "fixed_weight_model" | "layer_frozen_model"
    defines which of the three flavors of models to use (see get_new_model_helper)

  Returns
  -------
  A keras Sequential model
  """
  return get_new_model_helper(x_train.shape[1:], 2, activation_name, 16, flavor)

def get_new_model_helper(input_shape, output_shape, activation_name, layer_size, flavor=None):
 
  """
  Helper function for getting a new model

  ...


  Parameters
  ----------
  input_shape: Tuple
    a tuple of the shape of the input (if there are d features, this is (d,))
  output_shape: Tuple
    a tuple of the output shape (if there are x classes, this is (x,))
  activation_name: "custom_sigmoid" or "scaled_sigmoid" | "sign" | some TensorFlow activation name
    the name of the activation function to use
  layer_size: int
    the number of hidden layers
  flavor: "model" or None | "logit_model" | "fixed_weight_model" | "layer_frozen_model"
    which of the models to return (see code below)

  Returns
  -------
  a keras Sequential model
  """

  # find the activation function
  if activation_name == "custom_sigmoid" or activation_name == "scaled_sigmoid":
    activation = custom_sigmoid
  elif activation_name == "sign":
    activation = sign
  else:
    activation = keras.activations.deserialize(activation_name)

  # find which model to return and delegate
  if flavor is None or flavor == "model":
    model = get_model(input_shape, output_shape, activation, layer_size)
  elif flavor == "logit_model":
    model = get_logit_model(input_shape, output_shape, activation, layer_size)
  elif flavor == "fixed_weight_model":
    model = get_fixed_weight_model(input_shape, output_shape, activation, layer_size)
  elif flavor == "frozen_layer_model":
    model = get_frozen_layer_model(input_shape, output_shape, activation, layer_size)
  else:
    raise ValueError(f"invalid value {flavor} for argument flavor")

  return model


def load_mnist_model(x_train, y_train, file_name, layer_size, flavor=None): 
  """
  Loads model from a file that will classify the MNIST dataset

  ...

  Parameters
  ----------
  x_train: Numpy Array
    This is the MNIST training X set (used for dimensions)
  y_train: Numpy Array
    This is the MNIST training Y set (used for dimensions and to find number of classes)
  file_name: string in format {file name}:{activation function name}
    the part before the colon is the .h5 file that the weights are in
    the part after the colon is the name of the activation function to use
  layer_size: int
    the number of neurons in the hidden layer
  flavor: "model" (or None) | "logit_model" | "fixed_weight_model"
    defines which of the three flavors of models to use (see get_new_model_helper)

  Returns
  -------
  A keras Sequential model
  """
  return load_model_helper(x_train, y_train, file_name, layer_size, flavor)

def load_model(x_train, file_name, flavor=None):
  """
  Loads model from a file that will classify the PDFRateB or Hidost dataset

  ...

  Parameters
  ----------
  x_train: Numpy Array
    This is the MNIST training X set (used for dimensions)
  file_name: string in format {file name}:{activation function name}
    the part before the colon is the .h5 file that the weights are in
    the part after the colon is the name of the activation function to use
  flavor: "model" (or None) | "logit_model" | "fixed_weight_model" | "frozen_layer_model"
    defines which of the three flavors of models to use (see get_new_model_helper)

  Returns
  -------
  A keras Sequential model
  """
  return load_model_helper(x_train, None, file_name, 16, flavor)

def load_model_helper(x_train, y_train, file_name, layer_size, flavor=None): 
  """
  Loads model from a file

  ...

  Parameters
  ----------
  x_train: Numpy Array
    This is the MNIST training X set (used for dimensions)
  y_train: Numpy Array
    This is the MNIST training Y set (used for dimensions and to find number of classes)
  file_name: string in format {file name}:{activation function name}
    the part before the colon is the .h5 file that the weights are in
    the part after the colon is the name of the activation function to use
  layer_size: int
    the number of neurons in the hidden layer
  flavor: "model" (or None) | "logit_model" | "fixed_weight_model" | "frozen_layer_model" 
    defines which of the three flavors of models to use (see get_new_model_helper)

  Returns
  -------
  A keras Sequential model
  """
  colon_index = file_name.find(":")
  try:
    assert(colon_index > 0)
  except Exception as e:
    print("You must pass models in the format {file_name.h5}:{activation_name}")
    raise e

  weights_file = file_name[0:colon_index]
  activation_name = file_name[colon_index+1:]

  needs_reload = (flavor == "fixed_weight_model" or flavor == "frozen_layer_model")

  first_flavor = flavor

  if needs_reload:
    first_flavor = None

  if y_train is None:
    first_model = get_new_model(x_train, activation_name, flavor=first_flavor)
  else:
    first_model = get_new_mnist_model(x_train, y_train, activation_name, layer_size, first_flavor)

  first_model.build(x_train.shape)
  first_model.load_weights(weights_file)

  # get the weights out of the first model and into the model of the actual flavor we want
  # we have to do this because tf won't let us load weights directly into the partially frozen models
  if needs_reload:
    weights = first_model.get_weights()
  
    if y_train is None:
      model = get_new_model(x_train, activation_name, flavor=flavor)
    else:
      model = get_new_mnist_model(x_train, y_train, activation_name, layer_size, flavor)
 
    model.build(x_train.shape)
    model.set_weights(weights)

  else:
    model = first_model

  return model, activation_name


def sign(tensor):
  return tf.math.sign(tensor)





















  

import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow_probability as tfp

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

# this stabilizes weights in the l_1 case. Could be extended to work in other norms.
def stabilize_weights(layer, N = 100000):
  # generate the random data
  # NOTE: Can't Reuse Randomness

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  random_point = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  # classify the random data Crun through the model)
  output = layer.predict(random_point)

  # calculate the t_hats
  new_weights = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)

  # take the sign for the l_1 case
  new_weights = tf.math.sign(new_weights)

  # calculate the magnitude of the weight vector for each neuron
  new_mags = tf.norm(new_weights, ord=np.inf, axis=0)

  # get the weights of the old neuron
  old_weights = layer.get_weights()[0]

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=np.inf, axis=0)

  # calculate the proper scales 
  scales = old_mags / new_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  return scaled_weights
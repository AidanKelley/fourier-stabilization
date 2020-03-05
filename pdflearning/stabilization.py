import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow_probability as tfp

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

# this stabilizes weights in the l_1 case. Could be extended to work in other norms.
def stabilize_l1(layer):
  # generate the random data
  # NOTE: Can't Reuse Randomness

  input_shape = layer.input_shape

  # get the old weights from this layer
  old_weights = layer.get_weights()[0]

  # take the sign to get the new weights
  new_weights = tf.math.sign(old_weights)

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=np.inf, axis=0)

  # calculate the proper scales 
  scales = old_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  return scaled_weights
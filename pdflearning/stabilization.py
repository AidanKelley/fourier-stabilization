import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow_probability as tfp

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

def stabilize_l2(layer, N = 100000):
  p = 20
  p = float(p)
  assert(p > 1)

  q = 1/(1 - 1/p)
  print(f"q={q}")

  # generate the random data
  # NOTE: Can't Reuse Randomness

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  random_point = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  # classify the random data Crun through the model)
  output = layer.predict(random_point)

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)
  absolute_values = tf.math.pow(tf.math.abs(t_hats), p - 1)
  unscaled_weights = tf.math.multiply(tf.math.sign(t_hats), absolute_values)

  # get the weights of the old neuron
  old_weights = layer.get_weights()[0]

  # calculate the magnitude of the weights
  new_mags = tf.norm(unscaled_weights, ord=q, axis=0)
  old_mags = tf.norm(old_weights, ord=q, axis=0)

  # calculate the proper scales and scale the unscaled weights 
  scales = old_mags / new_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  new_weights = tf.linalg.matmul(unscaled_weights, scale_matrix)

  print(new_weights - old_weights)
  delta = new_weights - old_weights
  print(f"l_inf norm: {tf.norm(delta, ord=np.inf)}")
  print(f"l_{p} norm: {tf.norm(delta, ord=p)}")
  print(f"l_{q} norm: {tf.norm(delta, ord=q)}")

  print(f"l_{p} norm of old_weights: {tf.norm(old_weights, ord=p)}")
  print(f"l_{p} norm of new_weights: {tf.norm(new_weights, ord=p)}")
  
  print(f"l_{q} norm of old_weights: {tf.norm(old_weights, ord=q)}")
  print(f"l_{q} norm of new_weights: {tf.norm(new_weights, ord=q)}")

  print(f"l_inf norm ratio: {tf.norm(delta, ord=np.inf)/tf.norm(old_weights, ord=np.inf)}")
  print(f"l_{p} norm ratio: {tf.norm(delta, ord=p)/tf.norm(old_weights, ord=p)}")
  print(f"l_{q} norm ratio: {tf.norm(delta, ord=q)/tf.norm(old_weights, ord=q)}")

  return new_weights

# this stabilizes weights in the l_1 case. Could be extended to work in other norms.
def stabilize_l1(layer):
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
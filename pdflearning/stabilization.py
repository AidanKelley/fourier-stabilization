import tensorflow as tf
from tensorflow import keras

import numpy as np

import tensorflow_probability as tfp

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

def stabilize_lp(p, layer, codes=[], N = 1000000):
  p = float(p)
  assert(p > 1)

  q = 1/(1 - 1/p)
  print(f"p = {p}, q = {q}")

  # generate the random data
  # NOTE: Can't Reuse Randomness

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  random_point = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  # classify the random data (run through the model)
  output = layer.predict(random_point)

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)
  
  # if there are codes, include them here
  if codes is not None and len(codes) > 0:
    coded_t_hats = code_coefficients(layer, codes, test_data=random_point)
    t_hats = tf.concat([t_hats, coded_t_hats], axis=0)

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
  print(f"l_inf norm of difference: {tf.norm(delta, ord=np.inf)}")
  print(f"l_{p} norm of diffenence: {tf.norm(delta, ord=p)}")
  print(f"l_{q} norm of diffenence: {tf.norm(delta, ord=q)}")

  print(f"l_{p} norm of old_weights: {tf.norm(old_weights, ord=p)}")
  print(f"l_{p} norm of new_weights: {tf.norm(new_weights, ord=p)}")
  
  print(f"l_{q} norm of old_weights: {tf.norm(old_weights, ord=q)}")
  print(f"l_{q} norm of new_weights: {tf.norm(new_weights, ord=q)}")

  print(f"l_inf norm ratio: {tf.norm(delta, ord=np.inf)/tf.norm(old_weights, ord=np.inf)}")
  print(f"l_{p} norm ratio: {tf.norm(delta, ord=p)/tf.norm(old_weights, ord=p)}")
  print(f"l_{q} norm ratio: {tf.norm(delta, ord=q)/tf.norm(old_weights, ord=q)}")

  return new_weights

# this stabilizes weights in the l_1 case. Could be extended to work in other norms.
def stabilize_l1(layer, codes=[]):
  input_shape = layer.input_shape

  # get the old weights from this layer
  old_weights = layer.get_weights()[0]

  # take the sign to get the new weights
  new_weights = tf.math.sign(old_weights)

  # if there are codes, we need to calculate those, too.
  if codes is not None and len(codes) > 0:
    coded_coefs = code_coefficients(layer, codes)
    coded_weights = tf.math.sign(coded_coefs)
    new_weights = tf.concat([new_weights, coded_weights], axis=0)

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=np.inf, axis=0)

  # calculate the proper scales 
  scales = old_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  return scaled_weights

def code_coefficients(layer, codes, test_data=None, predictions=None, N=100000):

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  if test_data is None:
    test_data = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  if predictions is None:
    predictions = layer.predict(test_data)

  tensors = []

  for index, code in enumerate(codes):
    relevant_columns = tf.gather(test_data, code, axis=1)
    parity_codes = tf.math.reduce_prod(relevant_columns, axis=1)
    parity_codes = tf.reshape(parity_codes, (N, 1))

    # calculate the t_hats
    t_hats = 1/N * tf.linalg.matmul(parity_codes, predictions, transpose_a = True)

    tensors.append(t_hats)

  return tf.concat(tensors, axis=0)














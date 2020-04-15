import tensorflow as tf

import tensorflow_probability as tfp

from coding import apply_codes

import numpy as np

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

def search_codes(layer, N=100000):

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  test_data = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  output = layer.predict(test_data)

  # this is too slow
  # print("started calc")
  # parity_codes = apply_codes(test_data, [index_set])
  # print("stopped calc")

  # get the columns of the indices we care about
  relevant_columns = tf.gather(test_data, index_set, axis=1)
  parity_codes = tf.math.reduce_prod(relevant_columns, axis=1)
  parity_codes = tf.reshape(parity_codes, (N, 1))

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(parity_codes, output, transpose_a = True)
  heuristic = float(mean(t_hats))


  print(t_hats)

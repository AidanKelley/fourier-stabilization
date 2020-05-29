from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("--out_model", dest="out_model_file", action="store")
parser.add_argument("--out_codes", dest="out_codes_file", action="store")

args = parser.parse_args()

dataset = args.dataset
in_file = args.in_file
out_model_file = args.out_model_file
out_codes_file = args.out_codes_file

from .data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)

# all these imports take a while so we do them after we see that the data returns correctly

import tensorflow as tf
from tensorflow import keras

from .models import get_model

import numpy as np
import tensorflow_probability as tfp

from .coding import save_codes, all_combinations

import heapq

model = get_model(input_shape=x_train.shape[1:])
model.build(x_train.shape)
model.load_weights(in_file)
model.evaluate(x_test, y_test, verbose=2)

N = 100000

coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

layer = keras.models.Model(inputs = model.layers[0].input, outputs = model.layers[0].output)

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

print(f"trying all combinations with n={input_shape[1]} p={2}")
codes_to_try = all_combinations(input_shape[1], 2) + all_combinations(input_shape[1], 1)

sets = [[] for _ in codes_to_try]
heuristics = [0 for _ in sets]
all_t_hats = [[] for _ in sets]

for index, index_set in enumerate(codes_to_try):

  relevant_columns = tf.gather(test_data, index_set, axis=1)
  parity_codes = tf.math.reduce_prod(relevant_columns, axis=1)
  parity_codes = tf.reshape(parity_codes, (N, 1))

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(parity_codes, output, transpose_a = True)
  heuristic = float(tf.norm(t_hats, ord=1)) # compute the L_1 norm

  #print(f"{index_set}: {heuristic}, {t_hats}")

  heuristics[index] = (heuristic, index_set, t_hats)
  sets[index] = index_set
  all_t_hats[index] = t_hats

  if (index % 1000 == 0):
    print(f"{index} trials done")

largest = heapq.nlargest(135, heuristics, key=lambda entry: entry[0])

codes = [entry[1] for entry in largest]

if (out_codes_file):
  save_codes(codes, out_codes_file)
else:
  print(f"codes = {codes}")












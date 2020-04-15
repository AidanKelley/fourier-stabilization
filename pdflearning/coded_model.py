from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("in_model", action="store")
parser.add_argument("in_codes", action="store")
parser.add_argument("--out_model", dest="out_model_file", action="store")

args = parser.parse_args()

dataset = args.dataset
in_model_file = args.in_model
in_codes_file = args.in_codes
out_model_file = args.out_model_file

from data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)

from coding import code_inputs, load_codes, save_codes

codes = load_codes(in_codes_file)
print("coding datasets")
x_train_coded = code_inputs(x_train, codes)
x_test_coded = code_inputs(x_test, codes)

print("coding finished")

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from models import get_model
import numpy as np

# load the model
model = get_model(input_shape=x_train.shape[1:])
model.build(x_train.shape)
model.load_weights(in_model_file)
model.evaluate(x_test, y_test, verbose=2)

weights = model.get_weights()

first_layer_weights = weights[0]
first_layer_biases = weights[1]

print("recalculating t_hats")

# assume we are doing L1 for now
layer_width = len(first_layer_weights[0])
new_coded_weights = [[0 for _ in range(layer_width)] for _ in codes]

N = 1000000
coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

layer = keras.models.Model(inputs = model.layers[0].input, outputs = model.layers[0].output)

input_shape = layer.input_shape
sample_shape = list(input_shape)
sample_shape[0] = N

test_data = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

predictions = layer.predict(test_data)

for index, code in enumerate(codes):
  relevant_columns = tf.gather(test_data, code, axis=1)
  parity_codes = tf.math.reduce_prod(relevant_columns, axis=1)
  parity_codes = tf.reshape(parity_codes, (N, 1))

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(parity_codes, predictions, transpose_a = True)

  for j in range(layer_width):
    new_coded_weights[index][j] = float(t_hats[0][j])

# scale weights for l1 case
for col in range(layer_width):
  magnitude = abs(float(first_layer_weights[0][col]))

  for row in range(len(codes)):
    new_coded_weights[row][col] = magnitude * np.sign(new_coded_weights[row][col])

new_first_layer = tf.concat([weights[0], new_coded_weights], axis=0)

weights[0] = new_first_layer

# now, we have the new weights! make the new model now

coded_model = get_model(input_shape=(len(weights[0]),))
coded_model.build(x_train_coded.shape)
coded_model.set_weights(weights)
coded_model.evaluate(x_test_coded, y_test, verbose=2)

if out_model_file:
  coded_model.save_weights(out_model_file)
  codes_file_name = out_model_file.split(".")[0] + "_codes.json"
  save_codes(codes, codes_file_name)


















from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
# parser.add_argument("in", action="store", dest="in_file")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-n", dest="n_models", action="store")
parser.add_argument("-N", dest="sizes", action="append")

args = parser.parse_args()

dataset = args.dataset
out_file = args.out_file
trials = int(args.n_models)
sizes = [int(size) for size in args.sizes]

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from .data import get_data

from .models import get_model, sign
from .stabilization import stabilize_lp
import json

x_train, y_train, x_test, y_test = get_data(dataset)

total_ratios = None

for i in range(trials):
  model = get_model(input_shape=x_train.shape[1:])

  model.fit(x_train, y_train, epochs = 20)

  weights = model.get_weights()

  test_model = get_model(input_shape=x_train.shape[1:], activation=sign)
  test_model.build(x_train.shape)
  test_model.set_weights(weights)

  layer = keras.models.Model(inputs = test_model.layers[0].input, outputs = test_model.layers[0].output)

  new_weights = stabilize_lp(2, layer, N=E_size)
  old_weights = model.get_weights()[0]

  delta = new_weights - old_weights

  delta_norms = tf.norm(delta, ord=2, axis=0)
  old_norms = tf.norm(old_weights, ord=2, axis=0)

  delta_ratios = delta_norms / old_norms

  print(delta_ratios)

  if total_ratios is None:
    total_ratios = delta_ratios
  else:
    total_ratios = tf.concat([total_ratios, delta_ratios], axis=0)

data = total_ratios.numpy().tolist()

if out_file is not None:
  with open(out_file, "w") as out_handle:
    json.dump({"data": data}, out_handle)
else:
  print(f"data = {data}")















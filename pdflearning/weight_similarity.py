from argparse import ArgumentParser
import psutil

parser = ArgumentParser()
parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-N", dest="sizes", action="append")

args = parser.parse_args()

dataset = args.dataset
in_file = args.in_file
out_file = args.out_file
sizes = [int(size) for size in args.sizes]

from data import get_data

x_train, y_train, _, _ = get_data(dataset)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from models import load_mnist_model
from stabilization import stabilize_lp
import json

if "mnist" in dataset:
  model, _ = load_mnist_model(x_train, y_train, in_file, 1024)
else:
  exit("expected MNIST")

# we are only going to do the stabilization on this layer
layer = tf.keras.models.Model(inputs=model.layers[0].input,
                              outputs = model.layers[0].output)

old_weights = model.get_weights()[0]
print(old_weights.shape)
old_normalized, _ = tf.linalg.normalize(old_weights, ord=2, axis=0)
print(old_normalized.shape)

counts = []
data = []


for N in sizes:
  print(f"N={N}")
  new_weights = stabilize_lp(2, layer, codes=[], N=N)

  new_normalized, _ = tf.linalg.normalize(new_weights, ord=2, axis=0)

  product = tf.multiply(new_normalized, old_normalized)
  inner_products = tf.reduce_sum(product, axis=1)

  print(inner_products)

  similarities = inner_products.numpy().tolist()

  counts.append(N)
  data.append(similarities)

  with open(out_file, "w+") as out_handle:
    json.dump({"counts": counts, "data": data}, out_handle)

  print(f"N={N}, memory={psutil.virtual_memory()}")






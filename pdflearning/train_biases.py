from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("-o", dest="out_file", action="store")

args = parser.parse_args()

dataset = args.dataset
in_file = args.in_file
out_file = args.out_file

from data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from models import load_model, get_new_model, load_mnist_model, get_new_mnist_model

data_shape = x_train.shape[1:]

# load the original model so we can load then extract the weights

if dataset == "mnist":
  orig_model, activation = load_mnist_model(x_train, y_train, in_file, 1024)
  model = get_new_mnist_model(x_train, y_train, activation, 1024, "fixed_weight_model")
else:
  orig_model, activation = load_model(x_train, in_file)
  model = get_new_model(x_train, activation, "fixed_weight_model")

w = orig_model.get_weights()

# get the new model and set it to have the stabilized weights
# we have to make a new one since this was technical has 3 layers instead of 2

model.set_weights(w)

# evaluate before the training
model.evaluate(x_test, y_test, verbose=2)

# train the new biases
model.fit(x_train, y_train, epochs = 20)

# evaluate
model.evaluate(x_test, y_test, verbose=2)

model.summary()
orig_model.set_weights(model.get_weights())

orig_model.evaluate(x_test, y_test, verbose=2)
if out_file is not None:
  orig_model.save_weights(out_file)
  print(f"saved weights to {out_file}")

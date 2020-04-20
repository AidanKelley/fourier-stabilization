from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("-i", dest="in_file", action="store")
parser.add_argument("-a", dest="activation", action="store")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-e", dest="epochs", action="store", default="20")

args = parser.parse_args()

dataset = args.dataset
activation = args.activation
out_file = args.out_file
in_file = args.in_file
epochs = int(args.epochs)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from data import get_data

from models import get_new_model, get_new_mnist_model, load_model, load_mnist_model



x_train, y_train, x_test, y_test = get_data(dataset)

if in_file is None:
  if dataset == "mnist":
    model = get_new_mnist_model(x_train, y_train, activation, 1024)
  else:
    model = get_new_model(x_train, activation)
else:
  if dataset == "mnist":
    model, _ = load_mnist_model(x_train, y_train, in_file, 1024)
  else:
    model, _ = load_model(x_train, in_file)

model.fit(x_train, y_train, epochs = epochs)

model.evaluate(x_test, y_test, verbose=2)

if out_file is not None:
  print(f"model saved to {out_file}")
  model.save_weights(out_file)



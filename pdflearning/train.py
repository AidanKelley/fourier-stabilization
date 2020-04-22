from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("-i", dest="in_file", action="store")
parser.add_argument("-a", dest="activation", action="store")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-e", dest="epochs", action="store", default="20")
parser.add_argument("-c", dest="checkpoints", action="store", default="5")
parser.add_argument("-s", dest="status_file", action="store")

args = parser.parse_args()

dataset = args.dataset
activation = args.activation
out_file = args.out_file
in_file = args.in_file
epochs = int(args.epochs)
coarse = int(args.checkpoints)
status_file = args.status_file

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from data import get_data

from models import get_new_model, get_new_mnist_model, load_model, load_mnist_model
from status import save_status


x_train, y_train, x_test, y_test = get_data(dataset)

if in_file is None:
  if "mnist" in dataset:
    model = get_new_mnist_model(x_train, y_train, activation, 1024)
  else:
    model = get_new_model(x_train, activation)
else:
  if "mnist" in dataset:
    model, _ = load_mnist_model(x_train, y_train, in_file, 1024)
  else:
    model, _ = load_model(x_train, in_file)

# if coarse is 0, interpret as no coarsening
if coarse == 0:
  coarse = epochs

total_iter = int((epochs + coarse - 1)/coarse)
for i in range(total_iter):
  model.fit(x_train, y_train, epochs=coarse)

  if out_file is not None:
    print(f"model saved to {out_file}")
    model.save_weights(out_file)

  save_status(status_file, f"{(i+1) * coarse} {model.evaluate(x_test, y_test)}")

model.evaluate(x_test, y_test, verbose=2)

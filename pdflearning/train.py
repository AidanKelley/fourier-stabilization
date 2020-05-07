from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store", help="The dataset on which to train the model (see data.py for full list)")
parser.add_argument("-i", dest="in_file", action="store", help="Optional, an in_file. This is if you want to continue training an existing model. Must be given in {file_name}.h5:{activation_name} format")
parser.add_argument("-a", dest="activation", action="store", help="The activation function to train the model on. Only used if there is no in_file (meaning this is a new model)")
parser.add_argument("-o", dest="out_file", action="store", help="Out file to save the model to (in .h5 format)")
parser.add_argument("-e", dest="epochs", action="store", default="20", help="number of epochs to train model for")
parser.add_argument("-c", dest="checkpoints", action="store", default="5", help="Coarsening factor. The model will be saved to the output file every {c} epochs")
parser.add_argument("-s", dest="status_file", action="store", help="If selected, a status file will be updated every {c} epochs so you can monitor the process as it runs")
parser.add_argument("-b", dest="do_biases", action="store_true", help="If this option is given, the weights will be frozen and only biases will be trained. This option is ignored if there is no in_file")
parser.add_argument("-w", dest="freeze_weights", action="store_true", help="This will only freeze the weights in the very first layer, and nothing else, allowing you to re-train these after stabilization")

args = parser.parse_args()

dataset = args.dataset
activation = args.activation
out_file = args.out_file
in_file = args.in_file
epochs = int(args.epochs)
coarse = int(args.checkpoints)
status_file = args.status_file
do_biases = args.do_biases
freeze_weights = args.freeze_weights

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

  if do_biases:
    model_flavor="fixed_weight_model"
  elif freeze_weights:
    model_flavor="frozen_layer_model"
  else:
    model_flavor=None

  if "mnist" in dataset:
    model, _ = load_mnist_model(x_train, y_train, in_file, 1024, flavor=model_flavor)
  else:
    model, _ = load_model(x_train, in_file, flavor=model_flavor)

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

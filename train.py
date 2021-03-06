from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store", help="The dataset on which to train the model (see data.py for full list)")
parser.add_argument("-i", dest="in_file", action="store", help="Optional, an in_file. This is if you want to continue training an existing model. Must be given in {file_name}.h5:{activation_name} format")
parser.add_argument("-a", dest="activation", action="store", help="The activation function to train the model on. Only used if there is no in_file (meaning this is a new model)")
parser.add_argument("-o", dest="out_file", action="store", help="Out file to save the model to (in .h5 format)")
parser.add_argument("-e", dest="epochs", action="store", default="20", help="the max number of epochs to train model for")
parser.add_argument("-c", dest="checkpoints", action="store", default="5", help="Coarsening factor. The model will be saved to the output file every {c} epochs")
parser.add_argument("-s", dest="status_file", action="store", help="If selected, a status file will be updated every {c} epochs so you can monitor the process as it runs")
parser.add_argument("-b", dest="do_biases", action="store_true", help="If this option is given, the weights will be frozen and only biases will be trained. This option is ignored if there is no in_file")
parser.add_argument("-w", dest="freeze_weights", action="store_true", help="This will only freeze the weights in the very first layer, and nothing else, allowing you to re-train these after stabilization")
parser.add_argument("-m", dest="model_type", action="store", default=None, help="The type of the model to do. Can be linear")

args = parser.parse_args()

dataset = args.dataset
activation = args.activation
out_file_template = args.out_file
in_file = args.in_file
epochs = int(args.epochs)
coarse = int(args.checkpoints)
status_file = args.status_file
do_biases = args.do_biases
freeze_weights = args.freeze_weights
model_type = args.model_type

if do_biases:
  model_flavor = "fixed_weight_model"
elif freeze_weights:
  model_flavor = "frozen_layer_model"
else:
  model_flavor = None

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from src.data import get_data

from src.models import load_general_model
from src.status import save_status

# put headers on status_file
save_status(status_file, "epoch loss accuracy")

x_train, y_train, x_test, y_test = get_data(dataset)

model, fake_model = load_general_model(x_train, y_train, in_file, 16, model_flavor, model_type, activation)

# if coarse is 0, interpret as no coarsening
if coarse == 0:
  coarse = epochs

total_iter = int((epochs + coarse - 1)/coarse)

max_acc = 0
max_index = -1

# create temporary directory to save stuff to

import tempfile
import os

with tempfile.TemporaryDirectory() as dir:

  for i in range(total_iter):

    # if it's the last iteration make sure to not do too many epochs
    num_epochs = coarse
    if i == total_iter - 1:
      num_epochs = epochs - coarse * (total_iter - 1)

    model.fit(x_train, y_train, epochs=num_epochs)

    current_epochs = coarse * i + num_epochs

    if out_file_template is not None:
      # out_file = out_file_template.replace("{e}", str(current_epochs))

      out_file = os.path.join(dir, f"{current_epochs}.h5")

      if freeze_weights or do_biases:
        fake_model.set_weights(model.get_weights())
        fake_model.save_weights(out_file)
      else:
        model.save_weights(out_file)
      print(f"model saved to {out_file}")

    loss, acc = model.evaluate(x_test, y_test)

    if acc > max_acc:
      max_acc = acc
      max_index = current_epochs

    save_status(status_file, f"{current_epochs} {loss} {acc}")

  print(f"max accuracy at {max_index} epochs")
  save_status(status_file, f"max accuracy at {max_index} epochs")
  model.evaluate(x_test, y_test, verbose=2)

  best_model_file = os.path.join(dir, f"{max_index}.h5")

  os.replace(best_model_file, out_file_template)
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

from models import load_model, get_new_model

data_shape = x_train.shape[1:]

# load the original model so we can load then extract the weights
orig_model, activation = load_model(x_train, in_file)
w = orig_model.get_weights()

# get the new model and set it to have the stabilized weights
model = get_new_model(x_train, activation, "fixed_weight_model")
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

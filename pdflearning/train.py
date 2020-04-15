from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("-a", dest="activation", action="store")
parser.add_argument("-o", dest="out_file", action="store")

args = parser.parse_args()

dataset = args.dataset
activation = args.activation
out_file = args.out_file


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from data import get_data

from models import get_new_model



x_train, y_train, x_test, y_test = get_data(dataset)

model = get_new_model(x_train, activation)

model.fit(x_train, y_train, epochs = 20)

model.evaluate(x_test, y_test, verbose=2)

if out_file is not None:
	print(f"model saved to {out_file}")
	model.save_weights(out_file)



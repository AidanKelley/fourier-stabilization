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

# all these imports take a while so we do them after we see that the data returns correctly

import tensorflow as tf
from tensorflow import keras

from models import get_model, sign
from stabilization import stabilize_l1

model = get_model(input_shape=x_train.shape[1:])
model.build(x_train.shape)
model.load_weights(in_file)
model.evaluate(x_test, y_test, verbose=2)

weights = model.get_weights()
print(weights)
weights[1] = 0 * weights[1]
print(weights)
model.set_weights(weights)


model.evaluate(x_test, y_test, verbose=2)
if out_file is not None:
	model.save_weights(out_file)
	print(f"model saved to {out_file}")
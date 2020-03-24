from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("norm", action="store")
parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("-o", dest="out_file", action="store")

args = parser.parse_args()

norm = args.norm
dataset = args.dataset
in_file = args.in_file
out_file = args.out_file

from data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)

# all these imports take a while so we do them after we see that the data returns correctly

import tensorflow as tf
from tensorflow import keras

from models import get_model, sign
from stabilization import stabilize_l1, stabilize_l2

model = get_model(input_shape=x_train.shape[1:])
model.build(x_train.shape)
model.load_weights(in_file)
model.evaluate(x_test, y_test, verbose=2)

layer = keras.models.Model(inputs = model.layers[0].input, outputs = model.layers[0].output)

if norm == "l0" or norm == "l1":
	new_weights = stabilize_l1(layer)
elif norm == "l2":
	new_weights = stabilize_l2(layer)
else:
	exit(f"norm '{norm}' not ok")


weights = model.get_weights()
weights[0] = new_weights

# additionally, zero the biases
weights[1] = 0 * weights[1]

model.set_weights(weights)


model.evaluate(x_test, y_test, verbose=2)

if out_file is not None:
	model.save_weights(out_file)
	print(f"model saved to {out_file}")
















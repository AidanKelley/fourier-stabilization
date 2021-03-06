from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("norm", action="store")
parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("-c", dest="codes", action="store")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-m", dest="model_type", action="store")

args = parser.parse_args()

norm = args.norm
dataset = args.dataset
in_file = args.in_file
out_file = args.out_file
code_file = args.codes
model_type = args.model_type

from src.data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)

# all these imports take a while so we do them after we see that the data returns correctly

import tensorflow as tf
from tensorflow import keras

from src.models import load_model, load_mnist_model, sign, load_general_model
from src.stabilization import stabilize_l1, stabilize_lp

from src.coding import load_codes

# if "mnist" in dataset:
#   model, _ = load_mnist_model(x_train, y_train, in_file, 1024)
# else:
#   model, _ = load_model(x_train, in_file)

model, fake_model = load_general_model(x_train, y_train, in_file, 1024, None, model_type)


model.evaluate(x_test, y_test, verbose=2)
layer = keras.models.Model(inputs = model.layers[0].input, outputs = model.layers[0].output)

if code_file is not None:
  codes = load_codes(code_file)
else:
  codes = None

if norm == "l0" or norm == "l1":
  new_weights = stabilize_l1(layer, codes)
elif norm == "l2":
  new_weights = stabilize_lp(2, layer, codes)
else:
  exit(f"norm '{norm}' not ok")


weights = model.get_weights()
weights[0] = new_weights

# additionally, zero the biases
weights[1] = 0 * weights[1]

if code_file is not None:
  x_train, y_train, x_test, y_test = get_data(dataset + ":" + code_file)

model.set_weights(weights)

model.evaluate(x_test, y_test, verbose=2)

if out_file is not None:
  model.save_weights(out_file)
  print(f"model saved to {out_file}")














from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("in_file", action="store")
parser.add_argument("-o", dest="out_file", action="store")
parser.add_argument("-a", dest="accuracies", action="append")
# parser.add_argument("--no-acc", dest="no_acc", action="store_true", default=False)

args = parser.parse_args()

dataset = args.dataset
in_file = args.in_file
out_file = args.out_file
accuracies = args.accuracies

assert("{e}" in out_file)

from src.data import get_data

x_train, y_train, x_test, y_test = get_data(dataset)

# all these imports take a while so we do them after we see that the data returns correctly

import tensorflow as tf
from tensorflow import keras

from src.models import load_model, load_mnist_model, sign, load_general_model
from src.stabilization import stabilize_some_l1

# accuracies are sorted so that we can just "keep going" after hitting 1 threshold

acc_string_and_values = [(float(a), a) for a in accuracies]
acc_string_and_values.sort(key=lambda x: x[0], reverse=True)

model, _ = load_general_model(x_train, y_train, in_file, 1024, None, None)

print(acc_string_and_values)

for acc, acc_string in acc_string_and_values:

  # stabilize it
  model = stabilize_some_l1(model, x_test, y_test, thresh=acc, allowed_layers=(0, 2), no_accuracy=True)

  # save it
  out_file_name = out_file.replace("{e}", acc_string)

  model.save_weights(out_file_name)
  print(f"saved to {out_file_name}")










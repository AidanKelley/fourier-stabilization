from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("dataset", action="store")
parser.add_argument("-i", dest='in_files', action="append")
parser.add_argument("-n", "--trials", dest='trials', action="store", default=100)
parser.add_argument("-o", dest='out_file', action="store")
parser.add_argument("--all", dest="do_all", action="store_true")

args = parser.parse_args()

dataset = args.dataset
in_files = args.in_files
n_trials = int(args.trials)
out_file = args.out_file
do_all = args.do_all
  

from data import get_data

_, _, x_test, y_test = get_data(dataset)


import tensorflow as tf
from tensorflow import keras

import numpy as np

from models import get_logit_model

from attacks import l0_attack

import json

data_shape = x_test.shape[1:]

# pad infile names for prettiness

lens = [len(name) for name in in_files]
max_len = max(lens)

names = [name + " " * (max_len - len(name)) for name in in_files]


models = []

for in_file in in_files:
  model = get_logit_model(input_shape=data_shape)
  model.load_weights(in_file)
  models.append(model)

freqs = [[0 for _ in range(data_shape[0])] for _ in models]

if do_all or n_trials >= x_test.shape[0]:
  test_indices = range(x_test.shape[0])
else:
  np.random.seed(1)
  test_indices = np.random.randint(0, x_test.shape[0], size = n_trials)

for count, i in enumerate(test_indices):
  target = int(1 - y_test[i] + 0.5)
  x0 = x_test[i]

  for index, model in enumerate(models):
    norm, _ = l0_attack(x0, target, model)
    freqs[index][norm] += 1

  for index, freq in enumerate(freqs):
    print(f"{names[index]}: {freq[1:30]}")
if out_file is not None:

  out_obj = {'file_names': in_files, 'freq_data': freqs}
  with open(out_file, "w") as out_handle:
    json.dumps(out_obj, out_handle)

  print(f"saved to {out_file}")
else:
  print(f"all_freqs = {freqs}")



























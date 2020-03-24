from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("attack", action="store")
parser.add_argument("dataset", action="store")
parser.add_argument("-i", dest='in_files', action="append")
parser.add_argument("-n", "--trials", dest='trials', action="store", default=100)
parser.add_argument("-o", dest='out_file', action="store")
parser.add_argument("--all", dest="do_all", action="store_true")

args = parser.parse_args()

attack = args.attack
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
import foolbox
import json

data_shape = x_test.shape[1:]

# pad infile names for prettiness
lens = [len(name) for name in in_files]
max_len = max(lens)
names = [name + " " * (max_len - len(name)) for name in in_files]

# load the models
models = []
for in_file in in_files:
  model = get_logit_model(input_shape=data_shape)
  model.load_weights(in_file)
  models.append(model)

# choose the data to test with
# if needed, just use all the data
if do_all or n_trials >= x_test.shape[0]:
  test_indices = range(x_test.shape[0])
# otherwise, take a random sample
else:
  np.random.seed(1)
  test_indices = np.random.choice(x_test.shape[0], size=n_trials, replace=False)

# the custom version of JSMA that we coded
if attack == "custom_jsma": 

  # set up the data structure that holds frequencies
  freqs = [[0 for _ in range(data_shape[0])] for _ in models]

  # run the attack for every test example
  for count, i in enumerate(test_indices):
    target = int(1 - y_test[i] + 0.5)
    x0 = x_test[i]

    # run the attack for every model
    for index, model in enumerate(models):
      norm, _ = l0_attack(x0, target, model)
      freqs[index][norm] += 1

    for index, freq in enumerate(freqs):
      print(f"{names[index]}: {freq[0:30]}")

  # save the results
  if out_file is not None:
    out_obj = {'file_names': in_files, 'freq_data': freqs}
    with open(out_file, "w") as out_handle:
      json.dump(out_obj, out_handle)

    print(f"saved to {out_file}")
  else:
    print(f"all_freqs = {freqs}")

# the Carlini and Wagner attack, provided by foolbox
elif attack == "carlini":
  # some data
  x_rand = x_test[test_indices]
  y_rand = y_test[test_indices]

  # data structure to store the (unordered) norms
  min_norms = [[] for _ in models]

  for model_index, model in enumerate(models):

    # foolbox setup
    foolbox_model = foolbox.models.TensorFlowEagerModel(model, bounds=(-2, 2))

    # defaults to misclassification (untargeted)
    attack = foolbox.attacks.CarliniWagnerL2Attack(foolbox_model)

    x_adversarial = attack(x_rand, y_rand,
                           unpack = True,
                           binary_search_steps=10,
                           max_iterations = 100,
                           confidence = 0,
                           learning_rate = 0.01,
                           initial_const = 0.0001,
                           abort_early = True)

    norms = tf.norm(x_rand-x_adversarial, axis=1)

    # Is this messy? Yes. Does it work? Yes.
    min_norms[model_index] = norms.numpy().tolist()
  
  # save the results
  if out_file is not None:
    out_obj = {'file_names': in_files, 'min_norms': min_norms}
    with open(out_file, "w") as out_handle:
      json.dump(out_obj, out_handle)

    print(f"saved to {out_file}")
  else:
    print(f"min_norms = {freqs}")    

else:
  exit("invalid attack")



























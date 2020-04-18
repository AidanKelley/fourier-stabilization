from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("attack", action="store")
parser.add_argument("-d", dest='datasets', action="append")
parser.add_argument("-i", dest='in_files', action="append")
parser.add_argument("-n", "--trials", dest='trials', action="store", default=100)
parser.add_argument("-o", dest='out_file', action="store")
parser.add_argument("--all", dest="do_all", action="store_true")

args = parser.parse_args()

attack = args.attack
datasets = args.datasets
in_files = args.in_files
n_trials = int(args.trials)
out_file = args.out_file
do_all = args.do_all
  

assert(len(datasets) == len(in_files))

from data import get_data

big_data = [get_data(dataset) for dataset in datasets]

import tensorflow as tf
from tensorflow import keras

import numpy as np

from models import load_model

from attacks import l0_attack
import foolbox
import json

import eagerpy as ep


# pad infile names for prettiness
lens = [len(name) for name in in_files]
max_len = max(lens)
names = [name + " " * (max_len - len(name)) for name in in_files]

# load the models
models = []
data_size = big_data[0][2].shape[0]

max_data_dim = max([data[0].shape[1] for data in big_data])

for index, in_file in enumerate(in_files):
  x_test = big_data[index][2]
  x_train = big_data[index][0]
  assert(x_test.shape[0] == data_size)

  data_shape = x_test.shape[1:]
  print(f"index: {index} data_shape:{data_shape} in_file: {in_file}")
  model, _ = load_model(x_train, in_file, flavor="logit_model")
  models.append(model)


# choose the data to test with
# if needed, just use all the data
if do_all or n_trials >= data_size:
  test_indices = range(data_size)
# otherwise, take a random sample
else:
  np.random.seed(1)
  test_indices = np.random.choice(data_size, size=n_trials, replace=False)

# the custom version of JSMA that we coded
if attack == "custom_jsma": 

  # set up the data structure that holds frequencies
  freqs = [[0 for _ in range(max_data_dim)] for _ in models]

  # run the attack for every test example
  for count, i in enumerate(test_indices):

    # run the attack for every model
    for index, model in enumerate(models):

      _, _, x_test, y_test = big_data[index]

      target = int(1 - y_test[i] + 0.5)
      x0 = x_test[i]

      norm, _ = l0_attack(x0, target, model)
      freqs[index][norm] += 1

    for index, freq in enumerate(freqs):
      print(f"{names[index]}: {freq}")

    if count % 5 == 0 and out_file is not None:
      out_obj = {'file_names': in_files, 'freq_data': freqs}
      with open(out_file, "w") as out_handle:
        json.dump(out_obj, out_handle)

  # save the results
  if out_file is not None:
    out_obj = {'file_names': in_files, 'freq_data': freqs}
    with open(out_file, "w") as out_handle:
      json.dump(out_obj, out_handle)

    print(f"saved to {out_file}")
  else:
    print(f"all_freqs = {freqs}")

# the Carlini and Wagner attack, provided by foolbox
elif attack in ["carlini", "brendel"]:
  # data structure to store the (unordered) norms
  min_norms = [[] for _ in models]

  for model_index, model in enumerate(models):
    _, _, x_test, y_test = big_data[model_index]

    # some data 
    x_rand = x_test[test_indices]
    y_rand = y_test[test_indices]

    x_rand = tf.convert_to_tensor(x_rand, dtype=tf.float32)
    y_rand = tf.convert_to_tensor(y_rand, dtype=tf.int32)

    # foolbox setup
    foolbox_model = foolbox.models.TensorFlowModel(model, bounds=(-2, 2))

    if attack == "carlini":
      # defaults to misclassification (untargeted)
      carlini_attack = foolbox.attacks.CarliniWagnerL2Attack(foolbox_model)

      x_adversarial = carlini_attack(x_rand, y_rand,
                             unpack = True,
                             binary_search_steps=10,
                             max_iterations = 100,
                             confidence = 0,
                             learning_rate = 0.01,
                             initial_const = 0.0001,
                             abort_early = True)

      norms = tf.norm(x_rand-x_adversarial, axis=1)

    elif attack == "brendel":
      print(f"acc: {foolbox.accuracy(foolbox_model, x_rand, y_rand)}")

      brendel_attack = foolbox.attacks.L1BrendelBethgeAttack()

      x_adversarial, a, b = brendel_attack(foolbox_model, x_rand, y_rand, epsilons=270)

      print(a)
      print(b)

      norms = tf.norm(x_rand - x_adversarial, ord=1, axis=1)

      print(norms)
    else:
      exit(f"\"{attack}\" is not a valid attack name")

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



























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
import art

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

if attack == "custom_jsma": 
  for count, i in enumerate(test_indices):
    target = int(1 - y_test[i] + 0.5)
    x0 = x_test[i]

    for index, model in enumerate(models):
      norm, _ = l0_attack(x0, target, model)
      freqs[index][norm] += 1

    for index, freq in enumerate(freqs):
      print(f"{names[index]}: {freq[0:30]}")

elif attack == "carlini":
  
  x_rand = x_test[test_indices]
  y_rand = y_test[test_indices]

  print(x_rand[0])
  print(tf.norm(x_rand[0], ord=2))

  model = models[0]

  # todo: have to decide whether or not to clip here: Should we allow a value below -1 or above 1?
  tf.compat.v1.disable_eager_execution()
  wrapped_model = art.classifiers.KerasClassifier(model, use_logits=False)

  attack_obj = art.attacks.evasion.CarliniL2Method(classifier=wrapped_model,
                                                   targeted=False,
                                                   learning_rate=0.01,
                                                   binary_search_steps=25,
                                                   max_iter=50,
                                                   initial_const=0.0001,
                                                   confidence=0)
  x_examples = attack_obj.generate(x_rand)
  
  tf.compat.v1.enable_eager_execution()

  norms = tf.norm(x_rand-x_examples, axis=1)

  print(norms[0])
  print(norms[1])
  print(model.predict(x_examples))

else:
  exit("invalid attack")

if out_file is not None:

  out_obj = {'file_names': in_files, 'freq_data': freqs}
  with open(out_file, "w") as out_handle:
    json.dump(out_obj, out_handle)

  print(f"saved to {out_file}")
else:
  print(f"all_freqs = {freqs}")



























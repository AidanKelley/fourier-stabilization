from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("attack", action="store", help="the name of the attack to use")
parser.add_argument("-d", dest='datasets', action="append", help="the name of each dataset to be used. Multiple are used, but all must have the same number of datapoints. All models are only tested against inputs from their datasets, but if multiple models are given, all are tested on the same indices accross all datasets. Multiple datasets are allowed because we might wanted to apply some transformation to the data before some of the models classify it")
parser.add_argument("-i", dest='in_files', action="append", help="the model, in the format {model_file}.h5:{activation function name}")
parser.add_argument("-n", "--trials", dest='trials', action="store", default=100, help="the number of inputs to run on")
parser.add_argument("-o", dest='out_file', action="store", help="the output .json file to store data from the run.")
parser.add_argument("--all", dest="do_all", action="store_true", help="if this flag is supplied, all data points will be used")

# get args
args = parser.parse_args()
attack = args.attack
datasets = args.datasets
in_files = args.in_files
n_trials = int(args.trials)
out_file = args.out_file
do_all = args.do_all
  
# make sure each model has a database
assert(len(datasets) == len(in_files))

# see if we are using MNIST or not
using_mnist = False
if "mnist" in datasets[0]:
  using_mnist = True
  # sanity check
  for dataset in datasets:
    assert("mnist" in dataset)
else:
  # sanity check
  for dataset in datasets:
    assert("mnist" not in dataset)

from data import get_data
# get all the data
big_data = [get_data(dataset) for dataset in datasets]

import tensorflow as tf
import numpy as np
import foolbox
import json
import eagerpy as ep
from models import load_model, load_mnist_model
from attacks import l0_attack, l0_multiclass_attack

# pad infile names for prettiness
lens = [len(name) for name in in_files]
max_len = max(lens)
names = [name + " " * (max_len - len(name)) for name in in_files]

# load the models
models = []

# get the number of data points in the first dataset testing partition
num_data_points = big_data[0][2].shape[0]

# get the max number fo features as this is the absolute maximal robustness
max_data_dim = max([data[0].shape[1] for data in big_data])
print(f"max_data_dim: {max_data_dim}")

# get the models
for index, in_file in enumerate(in_files):
  # make sure that all have the same number of datapoints
  x_test = big_data[index][2]
  assert(x_test.shape[0] == num_data_points)
  data_shape = x_test.shape[1:]
  print(f"index: {index} data_shape:{data_shape} in_file: {in_file}")

  # load in the model
  x_train = big_data[index][0]
  if using_mnist:
    y_train = big_data[index][1]
    model, _ = load_mnist_model(x_train, y_train, in_file, 1024, flavor="logit_model")
  else:
    model, _ = load_model(x_train, in_file, flavor="logit_model")
  
  models.append(model)

# choose the data to test with
# if needed, just use all the data
if do_all or n_trials >= num_data_points:
  test_indices = range(num_data_points)
# otherwise, take a random sample
else:
  # seeded for consistency. TODO: remove this when doing the actual test
  np.random.seed(1)
  test_indices = np.random.choice(num_data_points, size=n_trials, replace=False)

min_norms = [[] for _ in models]

def save_norms():
  # save the results
  if out_file is not None:
    out_obj = {'file_names': in_files, 'min_norms': min_norms}
    with open(out_file, "w") as out_handle:
      json.dump(out_obj, out_handle)

    print(f"saved to {out_file}")
  else:
    print(f"min_norms = {min_norms}")    



# the custom version of JSMA that we coded
if attack == "custom_jsma": 

  # run the attack for every test example
  for count, test_index in enumerate(test_indices):

    # run the attack for every model
    for index, model in enumerate(models):

      _, _, x_test, y_test = big_data[index]
      
      if using_mnist:
        current_class = int(y_test[test_index])
        x0 = x_test[test_index]

        norm, _ = l0_multiclass_attack(x0, current_class, 10, model)

      else:
        target = int(1 - y_test[test_index] + 0.5)
        x0 = x_test[test_index]

        norm, _ = l0_attack(x0, target, model)
      min_norms.append(norm)
        
    save_norms()


# the Carlini and Wagner attack, provided by foolbox
elif attack in ["carlini", "brendel"]:
  # data structure to store the (unordered) norms

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

else:
  exit("invalid attack")

save_norms()
























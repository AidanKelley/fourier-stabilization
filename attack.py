from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("attack", action="store", help="the name of the attack to use")
parser.add_argument("-d", dest='datasets', action="append", help="the name of each dataset to be used. Multiple are used, but all must have the same number of datapoints. All models are only tested against inputs from their datasets, but if multiple models are given, all are tested on the same indices accross all datasets. Multiple datasets are allowed because we might wanted to apply some transformation to the data before some of the models classify it")
parser.add_argument("-i", dest='in_files', action="append", help="the model, in the format {model_file}.h5:{activation function name}")
parser.add_argument("-m", dest='model_types', action="append")
parser.add_argument("-n", "--trials", dest='trials', action="store", default=100, help="the number of inputs to run on")
parser.add_argument("-o", dest='out_file', action="store", help="the output .json file to store data from the run.")
parser.add_argument("--all", dest="do_all", action="store_true", help="if this flag is supplied, all data points will be used")
parser.add_argument("-c", dest="batch_size", action="store", help="defines the size of the batch of inputs if the brendel attack is used (custom_jsma always uses batch size 1)", default=0)
parser.add_argument("-t", dest="test_indices", action="append", help="for debugging, you can run the attack on certain input indices", default=[])

# get args
args = parser.parse_args()
attack = args.attack
datasets = args.datasets
in_files = args.in_files
model_types = args.model_types
n_trials = int(args.trials)
out_file = args.out_file
do_all = args.do_all
batch_size = int(args.batch_size)
test_indices = [int(i) for i in args.test_indices]

# make sure each model has a database
assert(len(datasets) == len(in_files))
assert(len(model_types) == len(in_files))


from src.data import get_data
# get all the data
big_data = [get_data(dataset) for dataset in datasets]

import tensorflow as tf
import numpy as np
import foolbox
import json
import eagerpy as ep
from src.models import load_model, load_mnist_model, load_general_model
from src.attacks import l0_attack, l0_multiclass_attack

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
  y_train = big_data[index][1]

  model, _ = load_general_model(x_train, y_train, in_file, 1024, "logit_model", model_types[index], None)
 
  print("done loading a model")
  models.append(model)

# choose the data to test with
# if needed, just use all the data
if len(test_indices) == 0:
  if do_all or n_trials >= num_data_points:
    test_indices = range(num_data_points)
  # otherwise, take a random sample
  else:
    # seeded for consistency. TODO: remove this when doing the actual test
    np.random.seed(1)
    test_indices = np.random.choice(num_data_points, size=n_trials, replace=False)

print(test_indices)

# if batch size is 0, this means we don't want any batches
if batch_size <= 0:
  batch_size = len(test_indices)

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
      current_class = int(y_test[test_index])
      x0 = x_test[test_index]

      norm, _ = l0_multiclass_attack(x0, current_class, np.amax(y_test) + 1, model, change_at_once=1)
      min_norms[index].append(norm)
      print(min_norms)
        
    save_norms()


# L1 Brendel Bethge Attack
elif attack == "brendel":
  # data structure to store the (unordered) norms

  for model_index, model in enumerate(models):
    x_train, y_train, x_test, y_test = big_data[model_index]


    # foolbox setup
    foolbox_model = foolbox.models.TensorFlowModel(model, bounds=(-2, 2))

    init_attack = foolbox.attacks.DatasetMinimizationAttack(distance=foolbox.distances.LpDistance(1))
    init_attack.feed(foolbox_model, x_train)

    brendel_attack = foolbox.attacks.L1BrendelBethgeAttack(init_attack=init_attack)

    epsilons = np.array([*range(100*max_data_dim)])*0.01
    epsilons_list = epsilons.tolist()

    # figure out how many batches we need (ceiling)
    num_batches = int((len(test_indices) + batch_size - 1)/batch_size)
    for batch_index in range(num_batches):
    
      # get the batch of indices
      lower_index = batch_index * batch_size
      upper_index = min((batch_index + 1) * batch_size, len(test_indices))
      batch_indices = test_indices[lower_index:upper_index]

      # get the data 
      x_rand = x_test[batch_indices]
      y_rand = y_test[batch_indices]

      x_rand = tf.convert_to_tensor(x_rand, dtype=tf.float32)
      y_rand = tf.convert_to_tensor(y_rand, dtype=tf.int32)

      criterion = foolbox.criteria.Misclassification(y_rand)
      advs, _, success = brendel_attack(foolbox_model, x_rand, criterion, epsilons=None)

      diffs = x_rand - advs

      norms = tf.norm(diffs, axis=1, ord=1)
      
      min_norms[model_index] += norms.numpy().tolist()
      save_norms()

else:
  exit("invalid attack")

save_norms()
























from sklearn import datasets
import numpy as np

import tensorflow as tf

from .coding import load_codes, code_inputs
from .gray_codes import do_gray_code, do_binary
from .uniform_coding import do_uniform_code

def cast_float(x):
  return x.astype(np.float32)

def cast_int(x):
  return x.astype(np.int32)

def create_partition(x_orig_train, y_orig_train, p_train=0.2):
  # inspired by pberkes answer: https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros, 
  # seed so we always get the same partition (can be changed later)
  np.random.seed(0)

  # generate random indices
  random_indices = np.random.permutation(x_orig_train.shape[0])

  # calculate how much to put in each partition
  test_size = int(x_orig_train.shape[0] * p_train)
  # split up the training and testing data in the same way
  testing_indices = random_indices[:test_size] # all before test_size
  training_indices = random_indices[test_size:] # all after test_size

  x_train, y_train = x_orig_train[training_indices, :], y_orig_train[training_indices]
  x_test, y_test = x_orig_train[testing_indices, :], y_orig_train[testing_indices]

  return x_train, y_train, x_test, y_test

def get_pdfrate(test=False):
  train_data = datasets.load_svmlight_file("pdf_dataset/data/pdfrateB_train.libsvm", n_features=135, zero_based=True)
  x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]

  x_train, y_train, x_test, y_test = create_partition(x_orig_train, y_orig_train)

  if test:
    test_data = datasets.load_svmlight_file("pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
    x_test, y_test = test_data[0].toarray(), test_data[1]
  
  x_train = 1 - 2*x_train
  x_test = 1 - 2*x_test
  
  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)

def get_hidost(test=False):
  train_data = datasets.load_svmlight_file("pdf_dataset/data/hidost_train.libsvm", n_features=961, zero_based=True)
  x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]

  x_train, y_train, x_test, y_test = create_partition(x_orig_train, y_orig_train)
 
  if test:
   test_data = datasets.load_svmlight_file("pdf_dataset/data/hidost_test.libsvm", n_features=961, zero_based=True)
   x_test, y_test = test_data[0].toarray(), test_data[1]

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)

def get_hatespeech(test=False):
  train_data = datasets.load_svmlight_file("hatespeech/gab_train.libsvm", n_features=200, zero_based=True)
  x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]

  x_train, y_train, x_test, y_test = create_partition(x_orig_train, y_orig_train)

  if test:
    test_data = datasets.load_svmlight_file("hatespeech/gab_test.libsvm", n_features=200, zero_based=True)
    x_test, y_test = test_data[0].toarray(), test_data[1]

  x_train = 1 - 2 * x_train
  x_test = 1 - 2 * x_test

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)

def get_fraud(test=False):
  train_data = datasets.load_svmlight_file("fraud/creditcard_train.libsvm", n_features=30, zero_based=True)
  x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]

  x_train, y_train, x_test, y_test = create_partition(x_orig_train, y_orig_train)

  if test:
    test_data = datasets.load_svmlight_file("fraud/creditcard_test.libsvm", n_features=30, zero_based=True)
    x_test, y_test = test_data[0].toarray(), test_data[1]

  x_train = 1 - 2 * x_train
  x_test = 1 - 2 * x_test

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)


def mnist_do_option(x_orig_train, option):
  if option is not None:
    if option == "gray":
      x_orig_train = do_gray_code(x_orig_train)
    elif option == "bin":
      x_orig_train = do_binary(x_orig_train)
    elif option == "bin2":
      x_orig_train = do_binary(x_orig_train/64, 2)
    elif option == "uniform":
      x_orig_train = do_uniform_code(x_orig_train, [255*(i+1)/9 for i in range(8)])
    elif option == "thresh":
      x_orig_train = do_uniform_code(x_orig_train, [127])
    elif option == "scaled":
      x_orig_train = x_orig_train.astype(np.float32) / 255
      x_orig_train = 1 - 2 * x_orig_train
    elif option == "masked":
      x_orig_train = mnist_do_option(x_orig_train, "thresh")
      # For Netanel: Implement this
      # generate a key (could be entirely random)
      # apply the key (xor to x_orig_train, make sure that's the variable being modified)
    else:
      exit(f"{option} is not a valid option for MNIST")

  return x_orig_train

def get_mnist(option=None, test=False):
  (x_orig_train, y_orig_train), (x_orig_test, y_orig_test) = tf.keras.datasets.mnist.load_data()
  # create a random partition to be used for testing -- don't touch the actual test data
  # make it consistent


  # flatten
  x_orig_train = mnist_do_option(x_orig_train, option)
  x_orig_train = x_orig_train.reshape((x_orig_train.shape[0], -1))

  # For Netanel: the code to display is x.reshape((28, 28)) to go from 784 to 28 by 28
  
  x_train, y_train, x_test, y_test = create_partition(x_orig_train, y_orig_train, p_train=1.0/6.0)

  print(x_train.shape)
  print(y_train.shape)

  if test:
    x_orig_test = mnist_do_option(x_orig_test, option)
    x_orig_test = x_orig_test.reshape((x_orig_test.shape[0], -1))

    x_test = x_orig_test
    y_test = y_orig_test

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)

def get_coded(original, code_file):
  codes = load_codes(code_file)
  new_x_train = code_inputs(original[0], codes)
  new_x_test = code_inputs(original[2], codes)

  return new_x_train, original[1], new_x_test, original[3]

def perform_restriction(data, restriction):
  x_train, y_train, x_test, y_test = data

  inverse_restriction_map = {digit: index for index, digit in enumerate(restriction)}

  def map_func(x):
    if x in inverse_restriction_map:
      return inverse_restriction_map[x]
    else:
      return - 1

  f = np.vectorize(map_func)


  train_conditions = f(y_train)
  train_indices = np.nonzero(train_conditions + 1)[0]

  test_conditions = f(y_test)
  test_indices = np.nonzero(test_conditions + 1)[0]

  return x_train[train_indices], train_conditions[train_indices], x_test[test_indices], test_conditions[test_indices]

def get_data(dataset):
  # see if we are doing the test set or not
  test_index = dataset.find("TEST_")
  test = False
  if test_index >= 0:
    dataset = dataset[len("TEST_"):]
    test = True

    print(f"Getting the testing dataset of {dataset} instead of the validation dataset")

  # see if we are doing codes or not
  colon_index = dataset.find(":")
  code_file = None

  if colon_index >= 0:
    code_file = dataset[colon_index + 1 : ]
    dataset = dataset[0:colon_index]

  print(f"dataset: {dataset} codes: {code_file}")

  # see if we should do a restriction or not
  restriction = None
  pipe_index = dataset.find(",")
  if pipe_index >= 0:
    restriction_str = dataset[pipe_index + 1:]
    restriction = [int(digit) for digit in restriction_str.split(",")]
    dataset = dataset[0:pipe_index]

  data = None
  if dataset == "pdfrate":
    data = get_pdfrate(test)
  elif dataset == "hidost":
    exit("there is an error if you are using raw hidost")
    # data = get_hidost(test)
  elif dataset == "hidost_scaled":
    x_train, y_train, x_test, y_test = get_hidost(test)
    data = (1 - 2 * x_train, y_train, 1 - 2 * x_test, y_test)
  elif dataset == "hatespeech":
    data = get_hatespeech(test)
  elif dataset == "fraud":
    data = get_fraud(test)
  elif dataset == "mnist":
    data = get_mnist(None, test)
  elif dataset == "mnist_gray":
    data = get_mnist("gray", test)
  elif dataset == "mnist_bin":
    data = get_mnist("bin", test)
  elif dataset == "mnist_bin2":
    data = get_mnist("bin2", test)
  elif dataset == "mnist_uniform":
    data = get_mnist("uniform", test)
  elif dataset == "mnist_thresh":
    data = get_mnist("thresh", test)
  elif dataset == "mnist_scaled":
    data = get_mnist("scaled", test)
  elif dataset == "mnist_masked":
    data = get_mnist("masked", test)
  else:
    quit("invalid dataset")

  # we want only the subset of the data that has y \in restrictions
  if restriction is not None:
    data = perform_restriction(data, restriction)

  if code_file is not None and len(code_file) > 0:
    data = get_coded(data, code_file)

  print(data[0].shape)
  print(data[2].shape)

  return data

from sklearn import datasets
import numpy as np
import tensorflow as tf

from coding import load_codes, code_inputs

def cast_float(x):
  return x.astype(np.float32)

def cast_int(x):
  return x.astype(np.int32)

def get_pdfrate():
  train_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_train.libsvm", n_features=135, zero_based=True)
  test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
  x_train, y_train = train_data[0].toarray(), train_data[1]
  x_test, y_test = test_data[0].toarray(), test_data[1]

  x_train = 1 - 2*x_train
  x_test = 1 - 2*x_test

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)


def get_hidost():
  train_data = datasets.load_svmlight_file("../pdf_dataset/data/hidost_train.libsvm", n_features=961, zero_based=True)
  # test_data = datasets.load_svmlight_file("data/hidost_test.libsvm", n_features=961, zero_based=True)
  x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]
  # X_test, y_test = test_data[0].toarray(), test_data[1]

  # inspired by pberkes answer: https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros, 

  # seed so we always get the same partition (can be changed later)
  np.random.seed(0)

  # generate random indices
  random_indices = np.random.permutation(x_orig_train.shape[0])

  # calculate how much to put in each partition

  test_size = int(x_orig_train.shape[0] / 5)

  # split up the training and testing data in the same way
  testing_indices = random_indices[:test_size] # all before test_size
  training_indices = random_indices[test_size:] # all after test_size

  x_train, y_train = x_orig_train[training_indices, :], y_orig_train[training_indices]
  x_test, y_test = x_orig_train[testing_indices, :], y_orig_train[testing_indices]

  return cast_float(x_train), cast_int(y_train), cast_float(x_test), cast_int(y_test)

def get_coded(original, code_file):
  codes = load_codes(code_file)
  new_x_train = code_inputs(original[0], codes)
  new_x_test = code_inputs(original[2], codes)

  return new_x_train, original[1], new_x_test, original[3]


def get_data(dataset):
  colon_index = dataset.find(":")
  code_file = None

  print(dataset)

  if colon_index >= 0:
    code_file = dataset[colon_index + 1 : ]
    dataset = dataset[0:colon_index]

  print(f"dataset: {dataset} codes: {code_file}")

  if dataset in ["pdfrate"]:
    data = get_pdfrate()
  elif dataset in ["hidost"]:
    data = get_hidost()
  else:
    quit("invalid dataset")

  if code_file is not None and len(code_file) > 0:
    data = get_coded(data, code_file)

  return data

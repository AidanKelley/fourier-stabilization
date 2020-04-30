from argparse import ArgumentParser

# set up the command line arguments
parser = ArgumentParser()
parser.add_argument("dataset", action="store", help="the dataset to use")
parser.add_argument("-i", dest="in_file", action="store", help="if given, we calculatethe similarity for this model")
parser.add_argument("-a", dest="activation", action="store", help="if wanting to train models, this is the activation function to use")
parser.add_argument("-o", dest="out_file", action="store", help="the (.json) output file where the data will be saved")
parser.add_argument("-N", dest="sizes", action="append", help="the size of the expectation to use. Multiple can be given by doing \" -N x -N y ... -N z\" ")
parser.add_argument("-n", dest="n_models", action="store", default="1", help="if wanting to train a model, this is the number of models that will be trained")

# load arguments into variables
args = parser.parse_args()

dataset = args.dataset
in_file = args.in_file
activation = args.activation
out_file = args.out_file
sizes = [int(size) for size in args.sizes]
n_models = int(args.n_models)

from data import get_data

# get the data (training only)
x_train, y_train, _, _ = get_data(dataset)

import tensorflow as tf
from models import load_mnist_model, get_new_mnist_model, get_new_model
from stabilization import stabilize_lp
import json
import psutil

# if there is an in_file, we only want to do one model
if in_file is not None:
  n_models = 1

# output arrays
counts = []
data = []

for i in range(n_models):
  if in_file is not None:
    if "mnist" in dataset:
      assert(n_models == 1)
      # load the weights for the model then move them into a new model
      model, _ = load_mnist_model(x_train, y_train, in_file, 1024)
      sign_model = get_new_mnist_model(x_train, y_train, "sign", 1024)
      sign_model.set_weights(model.get_weights())

    else:
      exit("expected MNIST")
  else:
    assert(activation is not None)
    assert("mnist" not in dataset)
    
    # get a new model and train it
    model = get_new_model(x_train, activation)
    model.fit(x_train, y_train, epochs=20)

    # change activation to sign for this to work right
    sign_model = get_new_model(x_train, "sign")
    sign_model.set_weights(model.get_weights())

  # we are only going to do the stabilization on this layer
  layer = tf.keras.models.Model(inputs=sign_model.layers[0].input,
                                outputs = sign_model.layers[0].output)

  # get and normalize the original weights
  old_weights = sign_model.get_weights()[0]
  old_normalized, _ = tf.linalg.normalize(old_weights, ord=2, axis=0)

  for size_index, N in enumerate(sizes):
    print(f"i={i}, N={N}")
    # get and normalize the new weights
    new_weights = stabilize_lp(2, layer, codes=[], N=N)
    new_normalized, _ = tf.linalg.normalize(new_weights, ord=2, axis=0)

    # calculate the inner product
    product = tf.multiply(new_normalized, old_normalized)
    inner_products = tf.reduce_sum(product, axis=0)

    # since both were normalized, inner product = cosine similarity
    similarities = inner_products.numpy().tolist()

    
    try:
      # combine the data of this size with the previous data
      data[size_index] += similarities
    except IndexError:
      # if the data does not already exist, create the array
      # and make an entry into counts
      counts.append(N)
      data.append(similarities)

    # save the data to a file every time there is new data
    with open(out_file, "w+") as out_handle:
      json.dump({"counts": counts, "data": data}, out_handle)

    # print memory utilization
    print(f"N={N}, memory={psutil.virtual_memory()}")






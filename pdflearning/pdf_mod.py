import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from pdf import get_model
from change_sign import get_sign_model

import tensorflow_probability as tfp

test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
x_test, y_test = test_data[0].toarray(), test_data[1]


x_test = 1 - 2*x_test

model = get_model()

model.evaluate(x_test, y_test, verbose=2)

model.load_weights("pdfmodel_weights.h5")

# just to make sure it really loaded in the weights
model.evaluate(x_test, y_test, verbose=2)

print(len(model.layers[0].get_weights()[0][0]))

# now, let's find those t_hats

def stabilize_weights(test_model, N = 100000):
  # generate the random data
  # NOTE: Can't Reuse Randomness
  coin_flip = tfp.distributions.Binomial(total_count = 1, probs = 0.5)
  random_point = 1 - 2 * coin_flip.sample(sample_shape=(N, 135))

  # classify the random data Crun through the model)
  output = test_model.predict(random_point)

  # calculate the t_hats
  new_weights = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)

  # take the sign for the l_1 case
  new_weights = tf.math.sign(new_weights)

  print(new_weights)

  # calculate the magnitue of the weight vector for each neuron
  new_mags = tf.norm(new_weights, ord=100, axis=0)

  # get the weights of the old neuron
  old_weights = test_model.get_weights()[0]

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=100, axis=0)

  # calculate the proper scales 
  scales = old_mags / new_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  # print out the ratio change just because
  changes = (scaled_weights - old_weights) / old_weights;
  print(changes)

  # now, modify the old weights to have the new weights for this layer
  old_total_weights = model.get_weights()
  old_total_weights[0] = scaled_weights

  # make a new model with the new, scaled weights
  new_model = get_sign_model()
  new_model.set_weights(old_total_weights)

  return new_model
  # first_new_input = new_weights[:, 0]


  # print(first_new_input)

test_model = keras.models.Model(inputs = model.input, outputs = model.layers[0].output)

new_model = stabilize_weights(test_model)
new_model.evaluate(x_test, y_test, verbose=2)

print(new_model.get_weights())

new_model.save_weights("pdfmodel_stabilized.h5");



















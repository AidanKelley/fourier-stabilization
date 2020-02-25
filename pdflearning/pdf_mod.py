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

coin_flip = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

# def calc_sign_expectation(index, weights, bias):

#   alpha = 0.01

#   for i in range(1, 10):
#     N = 1000 * (2**i)
#     sample_mean = calc_sample_mean(index, weights, bias, i**2)

#     alpha_i = alpha / (2**i)

#     critical = tf.math.sqrt(tf.math.log(alpha_i) / (2N)) #sqrt(ln alpha_i / 2N)

#     if sample_mean > critical or sample_mean < -1 * c:
#       return sign(sample_mean)

#   return 0

# def calc_sample_mean(index, weights, bias, N):
  
#   random_point = 1 - 2 * coin_flip.sample(sample_shape=(135, N))

#   classification = tf.math.sign(tf.linalg.matmul(weights, random_point) - bias)
#   sample_mean = tf.math.multiply(classification, random_point[index, :])

#   return -1

# now, let's find those t_hats

def stabilize_weights(test_model, N = 1000000, beta = 0.05):
  # generate the random data
  # NOTE: Can't Reuse Randomness
  random_point = 1 - 2 * coin_flip.sample(sample_shape=(N, 135))

  # classify the random data Crun through the model)
  output = test_model.predict(random_point)

  # calculate the t_hats
  new_weights = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)

  # take the sign for the l_1 case
  new_weights = (tf.math.sign(new_weights + beta) + tf.math.sign(new_weights - beta))/2
  # new_weights = tf.math.sign(new_weights)

  # calculate the magnitude of the weight vector for each neuron
  new_mags = tf.norm(new_weights, ord=np.inf, axis=0)

  # get the weights of the old neuron
  old_weights = test_model.get_weights()[0]

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=np.inf, axis=0)

  # calculate the proper scales 
  scales = old_mags / new_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  # print out the ratio change just because
  changes = (scaled_weights - old_weights) / old_weights;

  # now, modify the old weights to have the new weights for this layer
  old_total_weights = model.get_weights()
  old_total_weights[0] = scaled_weights

  # make a new model with the new, scaled weights
  new_model = get_sign_model()
  new_model.set_weights(old_total_weights)

  return new_model
  # first_new_input = new_weights[:, 0]


  # print(first_new_input)



alpha = 1 - np.math.pow(.95, 1/(135 * 16))

out_array = [[0 for _ in range(6)] for _ in range(3)]
variance_array = [[0 for _ in range(6)] for _ in range(3)]

test_model = keras.models.Model(inputs = model.input, outputs = model.layers[0].output)

sample_size = 10

for iter_n, n in enumerate([10**3, 10**4, 10**5]):
  
  beta = np.math.sqrt(-2 * np.math.log(alpha / 2) / n);
  print(f"beta: {beta}")


  for iter_beta, test_beta in enumerate([0.7 * beta, 0.9 * beta, beta, 1.1 * beta, 1.3 * beta, 0.05]):

    print(f"testing_beta: {beta}")
    for samples in range(sample_size):
      new_model = stabilize_weights(test_model, N = n, beta = test_beta)


      metric = new_model.evaluate(x_test, y_test, verbose=2)
      print(metric)
      out_array[iter_n][iter_beta] += metric[1]
      variance_array[iter_n][iter_beta] += metric[1] ** 2
      print(out_array)
      print(variance_array)

    out_array[iter_n][iter_beta] /= sample_size



#test_model = keras.models.Model(inputs = model.input, outputs = model.layers[0].output)

#new_model = stabilize_weights(test_model)
#new_model.evaluate(x_test, y_test, verbose=2)


#new_model.save_weights("pdfmodel_stabilized.h5");



















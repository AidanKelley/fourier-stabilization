import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from pdf import sign, custom_sigmoid, get_model
from change_sign import get_sign_model

import tensorflow_probability as tfp

def get_logit_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation=custom_sigmoid, input_shape=(135,)),
    keras.layers.Dense(2) # no activation function
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

def get_orig_logit_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation=custom_sigmoid, input_shape=(135,)),
    keras.layers.Dense(2) # no activation function
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
x_test, y_test = test_data[0].toarray(), test_data[1]


x_test = 1 - 2*x_test

x_test = tf.cast(x_test, tf.float32)

#just a test here

test_model = get_model()

test_model.load_weights("pdfmodel_stabilized.h5")

test_model.evaluate(x_test, y_test, verbose=2)


stable_model = get_logit_model()

stable_model.load_weights("pdfmodel_stabilized.h5")

orig_model = get_orig_logit_model()

orig_model.load_weights("pdfmodel_weights.h5")



def attack(x0, target, model):


  iter = 0

  n = tf.shape(x0)[0]

  x = x0
  
  all_eta = tf.ones((n, n)) -2 * tf.eye(n)

  not_target = 1 - target

  Zx_eta = model.predict(x_all_eta)

  log_target = Zx_eta[max_eta_index, target]
  log_not_target = Zx_eta[max_eta_index, not_target]

  while(iter < 10 and log_target < log_not_target):
    iter += 1

    x_2d = tf.reshape(x, (1, n))

    Zx = model.predict(x_2d)[0]

    x_mult = tf.tile(x_2d, (n, 1))

    x_all_eta = all_eta * x_mult

    Zx_eta = model.predict(x_all_eta)


    alpha = Zx_eta[:, target] - Zx[target]
    beta = Zx_eta[:, not_target] - Zx[not_target]

    alpha_mask = tf.math.greater(alpha, 0)
    beta_mask = tf.math.less(beta, 0)
    mask = tf.math.logical_and(alpha_mask, beta_mask)

    mask_float = tf.cast(mask, tf.float32)

    S = -1 * alpha * beta * mask_float

    max_eta_index = tf.math.argmax(S)

    x = all_eta[max_eta_index, :] * x

    log_target = Zx_eta[max_eta_index, target]
    log_not_target = Zx_eta[max_eta_index, not_target]

  return x, x - x0


total_stable = 0
total_orig = 0

print(stable_model.get_weights())
print(orig_model.get_weights())

count = 0

stable_freq = [0 for _ in range(20)]
orig_freq = [0 for _ in range(20)]

for i in np.random.randint(0, x_test.shape[0], size = 100):
  _, stable_eta = attack(x_test[i], int(1 - y_test[i] + 0.5), stable_model)
  _, orig_eta = attack(x_test[i], int(1 - y_test[i] + 0.5), orig_model)

  stable_norm = int(tf.norm(stable_eta, ord=1)/2 + 0.5)
  orig_norm = int(tf.norm(orig_eta, ord=1)/2 + 0.5)

  total_stable += stable_norm
  total_orig += orig_norm

  stable_freq[stable_norm] += 1
  orig_freq[orig_norm] += 1

  print(f"l_0 for stable: {stable_norm}, l_0 for orig: {orig_norm}")
  print(f"total_stable: {total_stable}, total_orig: {total_orig}")
  print(f"count: {count}")

  print(stable_freq)
  print(orig_freq)

  count += 1


print(f"total_stable: {total_stable}, total_orig: {total_orig}")











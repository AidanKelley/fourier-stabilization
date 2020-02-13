import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from pdf import sign, get_model

import tensorflow_probability as tfp

def get_sign_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation=sign, input_shape=(135,)),
    keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

if __name__ == '__main__':
  test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
  x_test, y_test = test_data[0].toarray(), test_data[1]


  x_test = 1 - 2*x_test

  model = get_sign_model()

  model.evaluate(x_test, y_test, verbose=2)

  model.load_weights("pdfmodel_weights.h5")

  # just to make sure it really loaded in the weights
  model.evaluate(x_test, y_test, verbose=2)


  sigmoid_model = get_model()

  sigmoid_model.load_weights("pdfmodel_weights.h5")

  sigmoid_model.evaluate(x_test, y_test, verbose=2)
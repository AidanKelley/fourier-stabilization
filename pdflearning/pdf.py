import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# get the data

from sklearn import datasets
train_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_train.libsvm", n_features=135, zero_based=True)
test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
x_train, y_train = train_data[0].toarray(), train_data[1]
x_test, y_test = test_data[0].toarray(), test_data[1]

# pre-process to so 1 goes to -1 and 0 to 1

x_train = 1 - 2*x_train
x_test = 1 - 2*x_test


# define the model

# lets do some fun stuff
# https://www.codespeedy.com/create-a-custom-activation-function-in-tensorflow/

# def sign(x):
#   return 1 if x >= 0 else -1

# np_sign = np.vectorize(sign)

# def d_sign(x):
#   return 1 if 0.5 <= x <= 0.5 else 0

# np_d_sign = np.vectorize(sign)

# np_sign_32 = lambda x: np_sign(x).astype(np.float32)

# print(np_sign_32([-1, -0.5, 9]))



# https://www.bignerdranch.com/blog/implementing-swish-activation-function-in-keras/

@tf.custom_gradient
def sign(tensor):

  def grad(dx):
    return dx

  return tf.math.sign(tensor), grad

def custom_sigmoid(tensor):

  return 2*tf.math.sigmoid(tensor) - 1


# the first category is 0 the second is 1
def get_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation=custom_sigmoid, input_shape=(135,)),
    keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

if __name__ == '__main__':

  model = get_model()

  model.fit(x_train, y_train, epochs=20)

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

  for layer in model.layers:
    print(f"config: {layer.get_config()}")

  model.save_weights("pdfmodel_weights.h5")






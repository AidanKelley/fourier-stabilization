import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from pdfrate_data import x_train, y_train, x_test, y_test

from models import get_model, get_fixed_weight_model

model = get_fixed_weight_model()
model.load_weights()
#model.build(input_shape=(135,))

model.summary()


model.set_weights(w)


model.evaluate(x_test, y_test, verbose=2)

model.fit(x_train, y_train, epochs = 20)

model.evaluate(x_test, y_test, verbose=2)


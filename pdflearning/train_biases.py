import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from pdfrate_data import x_train, y_train, x_test, y_test

from models import get_model, get_fixed_weight_model

# load the original model so we can load then extract the weights
orig_model = get_model()
orig_model.load_weights("pdfmodel_stabilized.h5")
w = orig_model.get_weights()

# get the new model and set it to have the stabilized weights
model = get_fixed_weight_model()
model.set_weights(w)

# evaluate before the training
model.evaluate(x_test, y_test, verbose=2)

# train the new biases
model.fit(x_train, y_train, epochs = 20)

# evaluate
model.evaluate(x_test, y_test, verbose=2)

model.summary()
orig_model.set_weights(model.get_weights())

orig_model.evaluate(x_test, y_test, verbose=2)

orig_model.save_weights("pdfmodel_trained_bias.h5")


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from pdfrate_data import x_train, y_train, x_test, y_test

from models import get_model

model = get_model()

model.fit(x_train, y_train, epochs = 20)

model.evaluate(x_test, y_test, verbose=2)

for layer in model.layers:
  print(f"config: {layer.get_config()}")

model.save_weights("pdfmodel_weights.h5")



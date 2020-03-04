import tensorflow as tf
from tensorflow import keras

from models import get_model, sign
from stabilization import stabilize_weights

from pdfrate_data import x_test, y_test

model = get_model(activation=sign)
model.load_weights("pdfmodel_weights.h5")
model.evaluate(x_test, y_test, verbose=2)

layer = keras.models.Model(inputs = model.layers[0].input, outputs = model.layers[0].output)

new_weights = stabilize_weights(layer)

weights = model.get_weights()
weights[0] = new_weights
model.set_weights(weights)


model.evaluate(x_test, y_test, verbose=2)
model.save_weights("sign_stabilized.h5")
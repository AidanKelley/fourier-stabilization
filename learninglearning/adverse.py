import tensorflow as tf
from tensorflow import keras

import numpy as numpy
import matplotlib.pyplot as plt

model = keras.models.load_model("fashion_model")

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocess

train_images = train_images / 255.0
test_images = test_images / 255.0

loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
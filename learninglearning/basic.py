import tensorflow as tf
from tensorflow import keras

import numpy as numpy
import matplotlib.pyplot as plt






def create_model():

  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

if __name__ == '__main__':

  print(tf.__version__)

  fashion_mnist = keras.datasets.fashion_mnist

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  # preprocess

  train_images = train_images / 255.0
  test_images = test_images / 255.0


# plt.figure(figsize=(5,5))
# for i in range(25):
#   plt.subplot(5, 5, i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(train_images[i], cmap=plt.cm.binary)
#   plt.xlabel(class_names[train_labels[i]])

# plt.show()

  model = create_model()

  model.fit(train_images, train_labels, epochs=10)

  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

  model.save('fashion_model')

  print(f'test accuracy: {test_acc}')





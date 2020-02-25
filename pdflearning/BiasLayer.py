from tensorflow.keras import layers
import tensorflow as tf

class BiasLayer(layers.Layer):

  def __init__(self, activation="linear"):
    super(BiasLayer, self).__init__()
    self.activation = activation

  def build(self, input_shape):
    self.b = self.add_weight(shape=(input_shape[-1],),
                             initializer='random_normal',
                             trainable=True)
    super(BiasLayer, self).build(input_shape)

  def call(self, inputs):
    return self.activation(inputs + self.b)
from tensorflow.keras import layers
import tensorflow as tf

class FixedWeight(layers.Layer):

  def __init__(self, out_dim, activation):
    super(FixedWeight, self).__init__()
    self.out_dim = out_dim
    self.activation = activation

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.out_dim),
                             initializer='zeros',
                             trainable=False)

    self.b = self.add_weight(shape=(self.out_dim,),
                             initializer='random_normal',
                             trainable=True)
    super(FixedWeight, self).build(input_shape)

  def call(self, inputs):
    inputs = tf.reshape(inputs, (1, 135))
    return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)

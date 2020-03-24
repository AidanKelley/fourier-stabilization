import tensorflow as tf
import tensorflow_probability as tfp

import json

d = 100
N = 1000000

model_distribution = tfp.distributions.Normal(0, 1)
coin_flip_distribution = tfp.distributions.Binomial(total_count = 1, probs = 0.5)

test_data = 1 - 2 * coin_flip_distribution.sample(sample_shape=(d, N))



def get_random_weights(size=d):
  weights = model_distribution.sample(sample_shape=(size, 1))
  magnitude = tf.norm(weights, ord=2)

  return weights/magnitude

def get_normed_t_hats(weights, bias=1):

  outputs = tf.math.sign(tf.linalg.matmul(weights, test_data, transpose_a = True) + bias)

  t_hats = 1.0/N * tf.linalg.matmul(test_data, outputs, transpose_b = True)

  magnitude = tf.norm(t_hats, ord=2)

  return t_hats / magnitude


trials = 10000

data = [0 for _ in range(trials)]

for i in range(trials):
  w = get_random_weights()
  new_w = get_normed_t_hats(w)

  dot = tf.linalg.matmul(w, new_w, transpose_a=True)

  float_dot = float(dot[0][0])
  print(float_dot)
  data[i] = float_dot

print(data)

with open("data_out_bias_big.json", "w") as out_file:
  json.dump({"data": data}, out_file)


















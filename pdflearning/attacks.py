import tensorflow as tf
import numpy as np

def l0_attack(x0, target, model):
  
  n = tf.shape(x0)[0]

  x = tf.cast(x0, tf.float32)
  
  all_eta = tf.ones((n, n)) -2 * tf.eye(n)

  not_target = 1 - target

  Zx_eta = model.predict(tf.reshape(x0, (1, n)))

  max_eta_index = 0
  log_target = Zx_eta[max_eta_index, target]
  log_not_target = Zx_eta[max_eta_index, not_target]

  iterations = 0

  flip_if_zero = n - 1;

  while log_target < log_not_target:
    iterations += 1
    # print(f"iterations: {iterations}, log_target: {log_target}, log_not_target: {log_not_target}")

    x_2d = tf.reshape(x, (1, n))

    Zx = model.predict(x_2d)[0]

    x_mult = tf.tile(x_2d, (n, 1))

    x_all_eta = all_eta * x_mult
    # print(x_all_eta)
    Zx_eta = model.predict(x_all_eta)


    alpha = Zx_eta[:, target] - Zx[target]
    beta = Zx_eta[:, not_target] - Zx[not_target]

    # print(f"Zx: {Zx}, Zx_eta:{Zx_eta}")

    # print(f"{alpha},{beta},")

    alpha_mask = tf.math.greater(alpha, 0)
    beta_mask = tf.math.less(beta, 0)
    mask = tf.math.logical_and(alpha_mask, beta_mask)

    # print(f"alpha_mask: {alpha_mask}, beta_mask: {beta_mask}")

    mask_float = tf.cast(mask, tf.float32)

    S = -1 * alpha * beta * mask_float

    
    max_eta_index = tf.math.argmax(S)

    if abs(S[max_eta_index]) < 10e-15:
      max_eta_index = flip_if_zero
      flip_if_zero -= 1


    print(f"iterations: {iterations} index: {max_eta_index} logits: {Zx} l0 norm:{tf.norm(x - x0, ord=1)/2}")

    # print(f"max: {max_eta_index}, S: {S}")

    x = all_eta[max_eta_index, :] * x

    log_target = Zx_eta[max_eta_index, target]
    log_not_target = Zx_eta[max_eta_index, not_target]



  l1_norm = tf.norm(x - x0, ord=1)
  l0_norm = int(l1_norm/2 + 0.5)

  return l0_norm, x - x0













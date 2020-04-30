import tensorflow as tf
import numpy as np

def l0_multiclass_attack(x0, orig_class, num_classes, model):
  x = tf.cast(x0, tf.float32)
  
  dim = tf.shape(x0)[0]
  flip_diag_mat = tf.ones((dim, dim)) - 2 * tf.eye(dim)

  target = 0
  other_indices = [*range(num_classes)]
  other_indices.remove(target)
  
  count = 0
  while count < 20:
    count += 1
  # # # compute forward derivative # # #
    x_2d = tf.reshape(x, (1, dim))
    print(x_2d)

    # get the initial prediction
    initial_logits = model.predict(x_2d)[0]
    print(initial_logits)

    # get the prediction after flipping each of the bits
    x_tiled = tf.tile(x_2d, (dim, 1))
    # note: this is element-wise multiplication and not matrix multiplication
    x_tiled_one_flipped = tf.multiply(flip_diag_mat, x_tiled)
    print(x_tiled_one_flipped)
    changed_logits = model.predict(x_tiled_one_flipped)
    print(changed_logits)

    # find "gradient", delta 
    gradient = (changed_logits - initial_logits)

  # # # compute the saliency map
    gradient_target = gradient[:, target]
    gradient_others = gradient[:, other_indices]
    sum_gradient_others = tf.reduce_sum(gradient_others, axis=1)

    print(sum_gradient_others)

    # we return zero if gradient target  < 0 or sum_gradient_others > 0
    # these are 0 if one of the conditions is violated
    target_mask = tf.math.greater(gradient_target, 0)
    others_mask = tf.math.less(sum_gradient_others, 0)
    # mask is 1 iff NOT(target < 0 or sum > 0)
    mask = tf.math.logical_and(target_mask, others_mask)
    float_mask = tf.cast(mask, tf.float32)

    # elementwise multiplication
    # multiply by 1 so that this is positive
    saliency_map = -1 * gradient_target * sum_gradient_others * float_mask
    
  # # # pick two points of max saliency
    _, maximal_indices = tf.math.top_k(saliency_map, k=100)

    print(maximal_indices)

    # for every index, flip that index in x
    print(x)
    for index in maximal_indices:
      x = flip_diag_mat[index, :] * x
    print(x)

  return 10000000000000, x - x0
  

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













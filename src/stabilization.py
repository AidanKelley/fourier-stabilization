import tensorflow as tf
import numpy as np

def stabilize_lp(p, layer, codes=[], N = 1000000):
  """
  Runs the Fourier Stabilization process in the l^p norm

  ...

  Parameters
  p: float > 1
    the norm to use
  layer
    the layer of the model that is getting stabilized
  codes: [[int]]
    potentially, a list of linear codes to use when doing the stabilization process
    this is a list of lists, where each of the inner lists is a set of indices
    then, this code is the linear polynomial on these indices
  N: int
    the number of terms to use in the approximation of each coefficient
  """
  
  # find the conjugate q norm
  p = float(p)
  assert(p > 1)

  q = p/(p-1)
  print(f"p = {p}, q = {q}, 1/p + 1/q = {1/p + 1/q}")

  # get the number of features
  input_shape = layer.input_shape
  features = input_shape[1]

  # generate the random data
  random_point = 1 - 2 * np.random.binomial(1, 0.5, (N, features)).astype(np.float32)

  # classify the random data (run through the model)
  output = layer.predict(random_point)

  print(f"output/predictions = {output}")

  # calculate the t_hats
  t_hats = 1/N * tf.linalg.matmul(random_point, output, transpose_a = True)
  
  # if there are codes, include them here
  if codes is not None and len(codes) > 0:
    coded_t_hats = code_coefficients(layer, codes, test_data=random_point)
    t_hats = tf.concat([t_hats, coded_t_hats], axis=0)

  absolute_values = tf.math.pow(tf.math.abs(t_hats), p - 1)
  unscaled_weights = tf.math.multiply(tf.math.sign(t_hats), absolute_values)

  # get the weights of the old neuron
  old_weights = layer.get_weights()[0]

  # calculate the magnitude of the weights
  new_mags = tf.norm(unscaled_weights, ord=q, axis=0)
  old_mags = tf.norm(old_weights, ord=q, axis=0)

  # calculate the proper scales and scale the unscaled weights 
  scales = old_mags / new_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  new_weights = tf.linalg.matmul(unscaled_weights, scale_matrix)

  print(new_weights - old_weights)
  delta = new_weights - old_weights
  print(f"l_inf norm of difference: {tf.norm(delta, ord=np.inf)}")
  print(f"l_{p} norm of diffenence: {tf.norm(delta, ord=p)}")
  print(f"l_{q} norm of diffenence: {tf.norm(delta, ord=q)}")

  print(f"l_{p} norm of old_weights: {tf.norm(old_weights, ord=p)}")
  print(f"l_{p} norm of new_weights: {tf.norm(new_weights, ord=p)}")
  
  print(f"l_{q} norm of old_weights: {tf.norm(old_weights, ord=q)}")
  print(f"l_{q} norm of new_weights: {tf.norm(new_weights, ord=q)}")

  print(f"l_inf norm ratio: {tf.norm(delta, ord=np.inf)/tf.norm(old_weights, ord=np.inf)}")
  print(f"l_{p} norm ratio: {tf.norm(delta, ord=p)/tf.norm(old_weights, ord=p)}")
  print(f"l_{q} norm ratio: {tf.norm(delta, ord=q)/tf.norm(old_weights, ord=q)}")

  return new_weights

# this stabilizes weights in the l_1 case. Could be extended to work in other norms.
def stabilize_l1(layer, codes=[]):
  input_shape = layer.input_shape

  # get the old weights from this layer
  old_weights = layer.get_weights()[0]

  # take the sign to get the new weights
  new_weights = tf.math.sign(old_weights)

  # if there are codes, we need to calculate those, too.
  if codes is not None and len(codes) > 0:
    coded_coefs = code_coefficients(layer, codes)
    coded_weights = tf.math.sign(coded_coefs)
    new_weights = tf.concat([new_weights, coded_weights], axis=0)

  # calculate the magnitude of the weights of the old neuron
  old_mags = tf.norm(old_weights, ord=np.inf, axis=0)

  # calculate the proper scales 
  scales = old_mags
  scale_matrix = tf.linalg.tensor_diag(scales)
  scaled_weights = tf.linalg.matmul(new_weights, scale_matrix)

  return scaled_weights

# compute the epsilons used in the "Loss of Accuracy" section of the paper
def compute_epsilons(layer):

  weights = layer.get_weights()[0]

  neuron_linfinity = tf.norm(weights, ord=np.inf, axis=0)
  neuron_l2 = tf.norm(weights, ord=2, axis=0)

  epsilons = neuron_linfinity / neuron_l2

  print(epsilons)
  return epsilons

def code_coefficients(layer, codes, test_data=None, predictions=None, N=100000):

  input_shape = layer.input_shape
  sample_shape = list(input_shape)
  sample_shape[0] = N

  if test_data is None:
    test_data = 1 - 2 * coin_flip_distribution.sample(sample_shape=sample_shape)

  if predictions is None:
    predictions = layer.predict(test_data)

  tensors = []

  for index, code in enumerate(codes):
    relevant_columns = tf.gather(test_data, code, axis=1)
    parity_codes = tf.math.reduce_prod(relevant_columns, axis=1)
    parity_codes = tf.reshape(parity_codes, (N, 1))

    # calculate the t_hats
    t_hats = 1/N * tf.linalg.matmul(parity_codes, predictions, transpose_a = True)

    tensors.append(t_hats)

  return tf.concat(tensors, axis=0)

# model is the model to be stabilized
# validation_x, validation_y are the x and y from the validation datasets
# threshhold is the minimum allowable accuracy,
# the algorithm greedily tries to maximize robustness
# while maintaining a minimum accuracy of the threshhold
# allowed_layers allows you to note multiple layers that can be stabilized.
# note that, however, that in tensorflow biases for one layer are their own layer.
# thus, if you wanted to allow the weights of the first and second layerss
# to be stabilized, you would set
# allowed layers to be (0, 2), with the 0 being the weights of the first layer
# and the 2 the weights of the 2nd.
# If the "fast" option is enabled, the algorithm uses an estimate for the change in
# accuracy which is not as good as the actual calculation. However, it is in fact very fast
def stabilize_some_l1(model, validation_x, validation_y, thresh=0.99, allowed_layers=(0,), no_accuracy=False):
  # iteratively stabilize a neuron with the highest "efficiency" but without breaking the threshhold
  # currently, the "fast" version, which I haven't really tested, will output a model just under the threshold

  def stabilize_neuron(layer_index, neuron_index, tentative=False):
    # if tentative, the change will only be made
    # if the accuracy threshhold is respected.

    current = model.get_weights()
    layer = current[layer_index]

    neuron_weights = layer[:, neuron_index]

    old_neuron_weights = None
    if tentative:
      old_neuron_weights = tf.identity(neuron_weights)

    neuron_l_inf = tf.norm(neuron_weights, ord=np.inf)
    new_neuron_weights = tf.math.sign(neuron_weights) * neuron_l_inf

    layer[:, neuron_index] = new_neuron_weights

    current[layer_index] = layer
    model.set_weights(current)

    if tentative:
      # if below the threshold, revert the change
      _, acc = model.evaluate(validation_x, validation_y)

      if acc < thresh:
        layer[:, neuron_index] = old_neuron_weights
        current[layer_index] = layer
        model.set_weights(current)

      return acc
    else:
      return None


  already_changed = set()

  should_continue = True

  # get current accuracy so we can compute the change in accuarcy
  _, current_accuracy = model.evaluate(validation_x, validation_y)

  # if we start and are already below the threshhold, we should return.
  if current_accuracy < thresh:
    should_continue = False

  while should_continue:
    # try out all the neurons everywhere
    best_neuron_layer = None
    best_neuron_index = None
    best_neuron_efficiency = None



    print(f"current acc: {current_accuracy}")



    for layer_index in allowed_layers:
      # first, compute some layer-specific numbers that will come in handy
      layer_weights = model.get_weights()[layer_index]
      l_inf_norms = tf.norm(layer_weights, ord=np.inf, axis=0)
      l1_norms = tf.norm(layer_weights, ord=1, axis=0)

      layer_size = layer_weights.shape[0]
      neuron_count = layer_weights.shape[1]
      print(f"layer_shape: {layer_weights.shape}")
      print(f"layer size: {layer_size}")

      # compute change in robustness for whole layer
      # after being stabilized in the l_1 way,
      # the weights will all be equal so the l1 norm wilil
      # just be equal to the layer size times the l_inf norm
      # then, our "robustness" will be equal to layer size
      # this is why layer size is involved in the calculation
      delta_robs = (layer_size - (l1_norms / l_inf_norms)).numpy()

      for neuron_index in range(neuron_count):
        if (layer_index, neuron_index) not in already_changed:
          accuracy_ok = True
          # compute the change in robustness
          delta_rob = delta_robs[neuron_index]

          # compute the change in accuracy
          if no_accuracy:
            efficiency = delta_rob
          else:
            # actually make the change to the model and see what happens
            stabilize_neuron(layer_index, neuron_index)

            # compute accuracy after this change
            _, model_accuracy = model.evaluate(validation_x, validation_y, verbose=0)

            if model_accuracy < thresh:
              accuracy_ok = False

            delta_acc = current_accuracy - model_accuracy

            # sometimes, the delta_acc may be 0, or even negative
            # we're going to correct this by just making it a
            # really small positive number instead
            # then, the algorithm will essentially rank on
            # just robustenss

            if delta_acc <= 0:
              delta_acc = 1e-20

            efficiency = delta_rob / delta_acc

          print(f"layer: {layer_index}, neuron: {neuron_index}, rob: {delta_rob}, acc: {'none' if no_accuracy else delta_acc}, ok: {accuracy_ok}")

          if accuracy_ok and (layer_index, neuron_index) not in already_changed:
            if best_neuron_efficiency is None or efficiency > best_neuron_efficiency:
              best_neuron_layer = layer_index
              best_neuron_index = neuron_index
              best_neuron_efficiency = efficiency

      # reset the model at the end of considering the current layer

      current_weights = model.get_weights()
      current_weights[layer_index] = layer_weights
      model.set_weights(current_weights)

    if best_neuron_efficiency is None:
      return model

    # otherwise, make the update

    assert(best_neuron_index is not None)
    assert(best_neuron_layer is not None)

    print(f"changing neuron at layer: {best_neuron_layer} index: {best_neuron_index}")

    current_accuracy = stabilize_neuron(best_neuron_layer, best_neuron_index, tentative=True)
    already_changed.add((best_neuron_layer, best_neuron_index))

  return model






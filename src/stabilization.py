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
def stabilize_some_l1(model, validation_x, validation_y, thresh=0.99, allowed_layers=(0,), no_accuracy=False, already_changed=None, verbo=2):
  # iteratively stabilize a neuron with the highest "efficiency" but without breaking the threshhold
  # currently, the "fast" version, which I haven't really tested, will output a model just under the threshold

  if already_changed is None:
    already_changed = set()

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
      _, acc = model.evaluate(validation_x, validation_y, verbose=verbo)
      did_change = True
      if acc < thresh:
        layer[:, neuron_index] = old_neuron_weights
        current[layer_index] = layer
        model.set_weights(current)
        did_change = False

      return acc, did_change
    else:
      return None


  total_delta_rob = 0
  should_continue = True

  # get current accuracy so we can compute the change in accuarcy
  _, current_accuracy = model.evaluate(validation_x, validation_y, verbose=verbo)

  # if we start and are already below the threshhold, we should return.
  if current_accuracy < thresh:
    should_continue = False

  while should_continue:
    # try out all the neurons everywhere
    best_neuron_layer = None
    best_neuron_index = None
    best_neuron_efficiency = None
    best_neuron_delta_rob = 0

    if verbo > 0:
      print(f"current acc: {current_accuracy}")

    for layer_index in allowed_layers:
      # first, compute some layer-specific numbers that will come in handy
      layer_weights = model.get_weights()[layer_index]
      l_inf_norms = tf.norm(layer_weights, ord=np.inf, axis=0)
      l1_norms = tf.norm(layer_weights, ord=1, axis=0)

      layer_size = layer_weights.shape[0]
      neuron_count = layer_weights.shape[1]

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

          # print(f"layer: {layer_index}, neuron: {neuron_index}, rob: {delta_rob}, acc: {'none' if no_accuracy else delta_acc}, ok: {accuracy_ok}")

          if accuracy_ok and (layer_index, neuron_index) not in already_changed:
            if best_neuron_efficiency is None or efficiency > best_neuron_efficiency:
              best_neuron_layer = layer_index
              best_neuron_index = neuron_index
              best_neuron_efficiency = efficiency
              best_neuron_delta_rob = delta_rob

      # reset the model at the end of considering the current layer

      current_weights = model.get_weights()
      current_weights[layer_index] = layer_weights
      model.set_weights(current_weights)

    if best_neuron_efficiency is None:
      return model, already_changed, total_delta_rob

    # otherwise, make the update

    assert(best_neuron_index is not None)
    assert(best_neuron_layer is not None)

    if verbo > 0:
      print(f"changing neuron at layer: {best_neuron_layer} index: {best_neuron_index}")

    current_accuracy, did_change = stabilize_neuron(best_neuron_layer, best_neuron_index, tentative=True)
    if did_change:
      already_changed.add((best_neuron_layer, best_neuron_index))
      total_delta_rob += best_neuron_delta_rob
    else:
      should_continue = False

  return model, already_changed, total_delta_rob

class stabilization_tracker:

  sorted_indices = []
  current_stabilized = 0 # all indices < than this one are stabilized, >= are not stabilized.

  def __init__(self, neuron_indices, heuristic):
    assert(len(neuron_indices) == len(heuristic))

    together = list(zip(neuron_indices, heuristic))
    together.sort(key=lambda x: x[1], reverse=True)

    self.sorted_indices = [x[0] for x in together]

  # return the indices that need to be stabilized
  def stabilize(self, up_to):
    assert(up_to >= self.current_stabilized)
    old = self.current_stabilized
    self.current_stabilized = up_to

    return self.sorted_indices[old : up_to]

  # return indices that need to be unstabilized
  def unstabilize(self, down_to):
    assert(down_to <= self.current_stabilized)
    old = self.current_stabilized
    self.current_stabilized = down_to

    return self.sorted_indices[down_to : old]

  def changed(self):
    return self.sorted_indices[:self.current_stabilized]

def stabilize_logn(model, validation_x, validation_y, thresh=0.99, allowed_layers=(0,), verbo=2):
  # algorithm basically works as follows (using binary searcH):
  # try stabilizing a certain number of neurons
  # If the accuracy is above the threshold, stabilize more
  # If it's below, stabilize fewer

  # The order of the stabilization is decided by descending order of the heuristic

  original_weights = [
    tf.identity(tensor)
    for tensor in model.get_weights()
  ]

  def calc_heuristic_for_layer(layer_weights):
    l_inf_norms = tf.norm(layer_weights, ord=np.inf, axis=0)
    l1_norms = tf.norm(layer_weights, ord=1, axis=0)
    heuristic = l1_norms/l_inf_norms
    np_array = heuristic.numpy()
    final_list = np_array.tolist()
    if isinstance(final_list, list):
      return final_list
    else:
      return [final_list]

  heuristic = [
    calc_heuristic_for_layer(layer_weights)
    for layer_weights in original_weights
  ]

  # now, make an array of form [(index in weights, heuristic value)]

  indices_and_heuristics = [
    ((layer_index, weight_index), heuristic_value)
    for layer_index, layer_heuristic in enumerate(heuristic)
    for weight_index, heuristic_value in enumerate(layer_heuristic)
  ]

  indices = [x[0] for x in indices_and_heuristics]
  heuristics = [x[1] for x in indices_and_heuristics]

  tracker = stabilization_tracker(indices, heuristics)

  # now, we just have to give an index, and the tracker will
  # tell us what to stabilize or unstabilize

  # Now, here is the binary search part
  upper_bound = len(indices) + 1
  lower_bound = 0
  last_index = 0
  next_index = int((upper_bound + lower_bound)/2)
  # we set the initial constants so that at the beginning
  # the loop will see that we need to stabilize
  # the first half of the indices

  should_continue = True
  while should_continue:

    print(f"next_index: {next_index}, last_index: {last_index}, u: {upper_bound}, l: {lower_bound}")

    if next_index > last_index:
      to_stabilize = tracker.stabilize(next_index)
      # stabilize these indices now
      current_weights = model.get_weights()
      for (layer_index, neuron_index) in to_stabilize:

        is_big = (len(current_weights[layer_index].shape) == 2)

        if not is_big:
          assert len(current_weights[layer_index].shape) == 1

        layer = current_weights[layer_index]

        if is_big:
          neuron_weights = layer[:, neuron_index]
        else:
          neuron_weights = layer[neuron_index]

        neuron_l_inf = tf.norm(neuron_weights, ord=np.inf)
        new_neuron_weights = tf.math.sign(neuron_weights) * neuron_l_inf

        if is_big:
          layer[:, neuron_index] = new_neuron_weights
        else:
          layer[neuron_index] = new_neuron_weights

        current_weights[layer_index] = layer

      model.set_weights(current_weights)

    elif next_index < last_index:
      to_unstabilize = tracker.unstabilize(next_index)

      current_weights = model.get_weights()
      for (layer_index, neuron_index) in to_unstabilize:
        if len(current_weights[layer_index].shape) == 1:
          current_weights[layer_index][neuron_index] = original_weights[layer_index][neuron_index]
        elif len(current_weights[layer_index].shape) == 2:
          current_weights[layer_index][:, neuron_index] = original_weights[layer_index][:, neuron_index]
        else:
          print(current_weights[layer_index])
          print("the shape was strange")
          assert(False)

      model.set_weights(current_weights)


    # evaluate the accuracy
    _, acc = model.evaluate(validation_x, validation_y, verbose=verbo)

    if acc < thresh:
      # move left
      if next_index == upper_bound:
        should_continue = False

      upper_bound = next_index
    else:
      # move right
      if next_index == lower_bound:
        should_continue = False

      lower_bound = next_index

    # shift indices
    last_index = next_index
    next_index = int((upper_bound + lower_bound) / 2)

  return model, tracker.changed(), None







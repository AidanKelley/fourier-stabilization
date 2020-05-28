import numpy as np

def do_uniform_code(x, cutoffs):
  before_concat = [threshold(x, thresh) for thresh in cutoffs]
  boolean_array = np.stack(before_concat, axis=-1)
  float_array = boolean_array.astype(np.float32)
  scaled_array = 1 - 2 * float_array

  return scaled_array


def threshold(x, thresh):
  binary_greater = np.greater(x, thresh)

  return binary_greater




if __name__ == "__main__":
  x_array = np.array([8*x for x in range(32)])
  print(do_uniform_code(x_array))

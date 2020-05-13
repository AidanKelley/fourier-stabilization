import numpy as np

# https://en.wikipedia.org/wiki/Gray_code
def select_bit(x, bit_index):
  x_shift = np.right_shift(x, bit_index);
  x_bit = np.bitwise_and(x_shift, 1)
  return x_bit

def to_binary(x, width):
  array_before_concat = [select_bit(x, bit) for bit in range(width)]

  return np.stack(array_before_concat, axis=-1)

def do_gray_code(x):
  x = x.astype(np.uint8)
  x_shift = np.right_shift(x, 1)
  gray_codes = np.bitwise_xor(x, x_shift)
  codes_binary = to_binary(gray_codes, 8)
  binary_floats = codes_binary.astype(np.float32)
  binary_scaled = 1 - 2 * binary_floats
  return binary_scaled

def do_binary(x, width=8):
  x = x.astype(np.uint8)
  x = to_binary(x, width)
  x = x.astype(np.float32)
  return 1 - 2 * x

if __name__ == "__main__":
  # do a little testing
  x_array = [[16 * (j + 4*i) for j in range(4)] for i in range(4)]
  x = np.array(x_array)
  print("hey")
  print(x.shape)
  print(x)
  coded = do_gray_code(x)
  print(coded.shape)
  print(coded)
  print(do_binary(x/64, 2))

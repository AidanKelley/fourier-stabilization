import json
import numpy as np

def save_codes(codes, file):
  with open(file, "w") as out_file:
    json.dump({"codes":codes}, out_file)

  print(f"codes saved to {file}")


def load_codes(file):
  with open(file, "r") as in_file:
    raw_codes = json.load(in_file)

  return raw_codes["codes"]

def apply_code(input_vector, code):
  product = 1

  for index in code:
    product *= input_vector[index]

  return product

def apply_codes(inputs, codes):
  num_codes = len(codes)

  coded = [
    [
      apply_code(input_vector, code) for code in codes
    ]
  for input_vector in inputs]

  return coded

def code_inputs(inputs, codes):
  coded = apply_codes(inputs, codes)
  return np.append(inputs, coded, axis=1)

def all_combinations(n, p):
  if p == 1:
    return [[i] for i in range(n)]
  else:
    all_combos = []
    for i in range(n-1):
      smaller_combos = all_combinations(n - i - 1, p - 1)
      combos_added = [[x + i + 1 for x in combo] for combo in smaller_combos]
      full_combos = [[i] + combo for combo in combos_added]
      all_combos += full_combos

    return all_combos 


import tensorflow as tf
from tensorflow import keras

import numpy as np

from pdfrate_data import x_test, y_test
from models import get_logit_model

from attacks import l0_attack



orig_model = get_logit_model()
orig_model.load_weights("pdfmodel_weights.h5")

stable_model = get_logit_model()
stable_model.load_weights("pdfmodel_trained_bias.h5")

n_trials = 1000

orig_freqs = [0 for _ in range(20)]
stable_freqs = [0 for _ in range(20)]

orig_accum = 0
stable_accum = 0

for count, i in enumerate(np.random.randint(0, x_test.shape[0], size = n_trials)):
  target = int(1 - y_test[i] + 0.5)
  x0 = x_test[i]

  orig_norm, _ = l0_attack(x0, target, orig_model)
  stable_norm, _ = l0_attack(x0, target, stable_model)

  orig_freqs[orig_norm] += 1
  stable_freqs[stable_norm] += 1

  orig_accum += orig_norm
  stable_accum += stable_norm

  print(orig_freqs)
  print(stable_freqs)

  print(f"orig: {float(orig_accum)/(count + 1)}, stable: {float(stable_accum)/(count + 1)}")


print("DONE!")
print(f"orig_freqs = {orig_freqs}")
print(f"stable_freqs = {stable_freqs}")



























from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-i", dest="in_files", action="append")
parser.add_argument("-b", dest="bins", action="store")

args = parser.parse_args()

in_files = args.in_files
try:
  bin_count = int(args.bins)
except TypeError:
  bin_count = 16


import json

data = [[] for _ in in_files]
for index, data_file in enumerate(in_files):
  with open(data_file, "r") as data_handle:
    data_json = json.load(data_handle)

  data[index] = data_json["data"]

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

# get consistent bins
all_data_together = np.hstack(data)
print(all_data_together)
# this generates the histogram, but just take the bin sizes (in [1])
bins = np.histogram(all_data_together, bins=bin_count)[1]

print(bins)

for index, hist in enumerate(data):
  plt.hist(hist, bins=bins)


plt.xlabel("$L^2$ norm of difference of normalized weights and normalized fourier coefficients")
plt.ylabel("frequency")

plt.legend()

plt.show()














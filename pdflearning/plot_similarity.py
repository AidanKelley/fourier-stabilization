from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--multi-in", dest="multi_in", action="store");
parser.add_argument("-i", dest="in_files", action="append")
parser.add_argument("-b", dest="bins", action="store")
parser.add_argument("-o", dest="out_dir", action="append")


args = parser.parse_args()

multi_in = args.multi_in
in_files = args.in_files
out_dir = args.out_dir

try:
  bin_count = int(args.bins)
except TypeError:
  bin_count = 16


import json

if multi_in is None:
  data = [[] for _ in in_files]
  labels = [[] for _ in in_files]
  for index, data_file in enumerate(in_files):
    with open(data_file, "r") as data_handle:
      data_json = json.load(data_handle)

    data[index] = data_json["data"]
    labels[index] = data_file

else:
  with open(multi_in, "r") as data_handle:
    data_json = json.load(data_handle)
    
  data = data_json["data"]
  labels = [multi_in + "_" + str(count) for count in data_json["counts"]]


from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

# get consistent bins
all_data_together = np.hstack(data)
print(all_data_together)
# this generates the histogram, but just take the bin sizes (in [1])
bins = np.histogram(all_data_together, bins=bin_count)[1]

if out_dir is None:
  print(bins)

  for index, hist in enumerate(data):
    plt.hist(hist, bins=bins, label=labels[index])


  plt.xlabel("$L^2$ norm of difference of normalized weights and normalized fourier coefficients")
  plt.ylabel("frequency")

  plt.legend()

  plt.show()

else:
 for index, series in data:
   series_string = "\n".join([str(num) for num in series])
   out_file = out_dir + "/" + labels[i] + ".txt"
   with open(out_file, "w+") as file_handle:
     file_handle.write("x\n")
     file_handle.write(series_string)












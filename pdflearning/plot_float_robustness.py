from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("data_file", action="store")
parser.add_argument("-o", dest="out_dir", action="store")
parser.add_argument("-s", dest="smooth", action="store_true")

args = parser.parse_args()

data_file = args.data_file
out_dir = args.out_dir
smooth = args.smooth

import json

with open(data_file, "r") as data_handle:
  data = json.load(data_handle)

min_norms = data["min_norms"]
names = data["file_names"]

from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np

max_val = max([max(min_norm) for min_norm in min_norms]) + 1

def make_hist(norms):
  norms.sort(reverse=True)

  x = norms
  y = np.linspace(0, 1, len(norms), endpoint=False).tolist()

  # append for 100% accuracy without any budget
  # however, this will probably be trimmed out

  x.append(0)
  y.append(1)

  cursor = 1

  # do some trimming
  while cursor < len(x):
    # if this is the same as the previous, remove it
    if x[cursor] == x[cursor-1]:
      x.pop(cursor)
      y.pop(cursor)
    # otherwise, increment cursor
    else:
      cursor += 1

  do_bumps = not smooth
  # do some processing for accuracy
  if do_bumps:
    cursor = 0
    while cursor < len(x):

      x.insert(cursor+1, x[cursor])
      y.insert(cursor+1, min(y[min(cursor+1, len(y) - 1)], 1))

      cursor += 2

  # add in a 0 at the end

  x.insert(0, max_val)
  y.insert(0, 0)

  # reverse both since we initially did this in reverse order
  x.reverse()
  y.reverse()

  return x, y

if out_dir is not None:
  file_names = data["file_names"]
  for index, norms in enumerate(min_norms):
    x, y = make_hist(norms)
    coordinates = "".join([f"{x[i]} {y[i]}\n" for i, _ in enumerate(x)])
    out_file = out_dir + "/" + names[index] + ".txt"
    with open(out_file, "w") as file_handle:
      file_handle.write("x y\n")
      file_handle.write(coordinates)

else:
  for index, norms in enumerate(min_norms):
    x, y = make_hist(norms)
    plt.plot(x, y, '-', label=names[index])

  plt.legend()

  plt.xlabel("$||\\eta|| \\leq x$")
  plt.ylabel("accuracy")

  plt.show()











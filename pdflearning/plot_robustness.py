from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("data_file", action="store")
parser.add_argument("-c", dest="clip", action="store")

args = parser.parse_args()

data_file = args.data_file

import json

with open(data_file, "r") as data_handle:
	data = json.load(data_handle)

all_freqs = data["freq_data"]

try:
	clip = int(args.clip)
except TypeError:
	clip = len(all_freqs[0])

from matplotlib import pyplot as plt
import matplotlib as mpl

all_freqs = [freq[:clip] for freq in all_freqs]

def make_hist(freqs):
  # integrate
  hist_misclass = [sum(freqs[0:i+1]) for i in range(len(freqs))]
  trials = hist_misclass[-1]
  hist = [float(trials - misclass)/trials for misclass in hist_misclass]
  return hist

all_hists = [make_hist(freq) for freq in all_freqs]

domain = [i for i in range(len(all_hists[0]))]

labels = ["original", "stabilized (zero biases)", "trained biases", "zero biases"]

for index, hist in enumerate(all_hists):
	plt.plot(domain, hist, '.-', label=labels[index])

plt.legend()

plt.xlabel("$||\\eta||_0 \\leq x$")
plt.ylabel("accuracy")

plt.suptitle("Accuracy on Hidost data following JSMA attack with limited budget ($n$ = 100)")

plt.show()














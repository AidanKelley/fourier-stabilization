from matplotlib import pyplot as plt
import matplotlib as mpl

orig_freqs = [0, 25, 36, 19, 87, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
stable_freqs = [1, 6, 10, 11, 87, 25, 20, 9, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

display_range = 12

orig_freqs = orig_freqs[0:display_range]
stable_freqs = stable_freqs[0:display_range]

def make_hist(freqs):
  # integrate
  hist_misclass = [sum(freqs[0:i+1]) for i in range(len(freqs))]
  trials = hist_misclass[-1]
  hist = [float(trials - misclass)/trials for misclass in hist_misclass]
  return hist

orig_hist = make_hist(orig_freqs)
stable_hist = make_hist(stable_freqs)

print(orig_hist)
print(stable_hist)

domain = [i for i in range(len(orig_hist))]

plt.plot(domain, orig_hist, label="original")
plt.plot(domain, stable_hist, label="stabilized")

plt.legend()

plt.xlabel("$||\\eta||_0 \\leq x$")
plt.ylabel("accuracy")

plt.suptitle("Accuracy vs. $l_0$ norm attack budget (n = 1000)")

plt.show()
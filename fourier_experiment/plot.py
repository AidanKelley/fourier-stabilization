from matplotlib import pyplot as plt
import json

with open("data_out_big.json", "r") as data_file:
  data_obj = json.load(data_file)


data1 = data_obj["data"]

with open("data_out.json", "r") as data_file:
  data_obj = json.load(data_file)

data2 = data_obj["data"]

plt.hist(data1)
plt.hist(data2)

plt.show()
filename_in = "fraud/creditcard.csv"
filename_train = "fraud/creditcard_train.libsvm"
filename_test = "fraud/creditcard_test.libsvm"

import statistics
import csv
import random
from imblearn.over_sampling import SMOTE
import numpy as np


with open(filename_in, "r") as in_file:
    rows = []
    classes = []

    reader = csv.reader(in_file)

    for index, line in enumerate(reader):
        if index != 0:
            rows.append(line[0:-1])
            classes.append(line[-1])

print("read the file")

X = np.array(rows, dtype=np.float32)
y = np.array(classes, dtype=np.int)

X_data, y_data = SMOTE().fit_resample(X, y)

print("dide the smote")

print(X_data.shape)

medians = np.median(X_data, axis=0)

almost_normal = X_data - medians

binarized = np.sign(almost_normal)

data_to_write = list(binarized)

with open(filename_train, "w") as train_file:
    with open(filename_test, "w") as test_file:
        for line_index, line in enumerate(data_to_write):
            line_out_array = [f"{index}:1" for index, x in enumerate(line)
                              if x > 0]

            line_out = str(y_data[line_index]) + " " + " ".join(line_out_array) + "\n"

            if random.randint(1, 2) == 1:
                train_file.write(line_out)
            else:
                test_file.write(line_out)

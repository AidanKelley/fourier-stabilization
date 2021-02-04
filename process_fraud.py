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
            # append if it is fradulent or with 10% chance otherwise
            if int(line[-1]) == 1 or random.randint(1, 10) == 10:
                rows.append(line[0:-1])
                classes.append(line[-1])

print("read the file")

X = np.array(rows, dtype=np.float32)
y = np.array(classes, dtype=np.int)

X_data, y_data = SMOTE().fit_resample(X, y)

# partition now
from src.data import create_partition
x_train, y_train, x_test, y_test = create_partition(X_data, y_data, p_train=0.5)

medians = np.median(x_train, axis=0)

x_train_medianed = x_train - medians
x_test_medianed = x_test - medians

def write_to_file(x_out, y_out, filename):
    data_to_write = list(x_out)

    with open(filename, "w") as out_file:
        for line_index, line in enumerate(data_to_write):
            line_out_array = [f"{index}:1" for index, x in enumerate(line)
                              if x > 0]

            line_out = str(y_out[line_index]) + " " + " ".join(line_out_array) + "\n"
            out_file.write(line_out)

write_to_file(x_train_medianed, y_train, filename_train)
write_to_file(x_test_medianed, y_test, filename_test)

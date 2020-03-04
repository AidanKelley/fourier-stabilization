from sklearn import datasets
import numpy as np

train_data = datasets.load_svmlight_file("../pdf_dataset/data/hidost_train.libsvm", n_features=961, zero_based=True)
# test_data = datasets.load_svmlight_file("data/hidost_test.libsvm", n_features=961, zero_based=True)
x_orig_train, y_orig_train = train_data[0].toarray(), train_data[1]
# X_test, y_test = test_data[0].toarray(), test_data[1]

# inspired by pberkes answer: https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros, 

# seed so we always get the same partition (can be changed later)
np.random.seed(1)

# generate random indices
random_indices = np.random.permutation(x_orig_train.shape[0])

# calculate how much to put in each partition

test_size = int(x_orig_train.shape[0] / 5)

# split up the training and testing data in the same way
training_indices = random_indices[:test_size] # all before test_size
testing_indices = random_indices[test_size:] # all after test_size

x_train, y_train = x_orig_train[training_indices, :], y_orig_train[training_indices]
x_test, y_test = x_orig_train[testing_indices, :], y_orig_train[testing_indices]




from sklearn import datasets
train_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_train.libsvm", n_features=135, zero_based=True)
test_data = datasets.load_svmlight_file("../pdf_dataset/data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
x_train, y_train = train_data[0].toarray(), train_data[1]
x_test, y_test = test_data[0].toarray(), test_data[1]

x_train = 1 - 2*x_train
x_test = 1 - 2*x_test


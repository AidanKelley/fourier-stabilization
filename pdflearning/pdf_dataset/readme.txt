1. Two datasets are provided, each with 10,082 training data and 9,771 test data.

2. The Hidost dataset uses 961 binary features corresponding to structual information of PDF files; PDFRate-B uses 135 binarized features in response to cotent information of PDF files.  

3. The datasets are with the libsvm format like "<label> <index1>:<value1> <index2>:<value2> ...". The malicious PDF documents are labeled as 1s, and the benigns are labeled as 0s. The indexes are in response to non-zero features.

4. Please install the sklearn library before loading the dataset. Specifically, use the following to load Hidost in Python:

from sklearn import datasets
train_data = datasets.load_svmlight_file("data/hidost_train.libsvm", n_features=961, zero_based=True)
test_data = datasets.load_svmlight_file("data/hidost_test.libsvm", n_features=961, zero_based=True)
X_train, y_train = train_data[0].toarray(), train_data[1]
X_test, y_test = test_data[0].toarray(), test_data[1]

Similarly, PDFRate-B can be loaded as follows:

from sklearn import datasets
train_data = datasets.load_svmlight_file("data/pdfrateB_train.libsvm", n_features=135, zero_based=True)
test_data = datasets.load_svmlight_file("data/pdfrateB_test.libsvm", n_features=135, zero_based=True)
X_train, y_train = train_data[0].toarray(), train_data[1]
X_test, y_test = test_data[0].toarray(), test_data[1] 

You will get Numpy arrays from the steps above. Good luck!
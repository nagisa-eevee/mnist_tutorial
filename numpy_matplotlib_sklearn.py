import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC


# download and load mnist data from https://www.openml.org/d/554
# for this tutorial, the data have been downloaded already in './scikit_learn_data'
X, Y = fetch_openml('mnist_784', data_home='./scikit_learn_data', return_X_y=True)
# print('fetch_openml() runs successfully.')

# make the value of pixels from [0, 255] to [0, 1] for further process
X = X / 255.

# split data to train and test (for faster calculation, just use 1/10 data)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

# Q1:Please use the logistic regression(default parameters) in sklearn
# to classify the data above, and print the training accuracy and test accuracy.
# TODO:use logistic regression
clf = LogisticRegression().fit(X_train, Y_train)

train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

print('Q1')
print('Training accuracy: %0.2f%%' % (train_accuracy * 100))
print('Testing accuracy: %0.2f%%' % (test_accuracy * 100))

# Q2:Please use the naive bayes(Bernoulli, default parameters) in sklearn
# to classify the data above, and print the training accuracy and test accuracy.
# TODO:use naive bayes
clf = BernoulliNB().fit(X_train, Y_train)

train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

print('Q2')
print('Training accuracy: %0.2f%%' % (train_accuracy * 100))
print('Testing accuracy: %0.2f%%' % (test_accuracy * 100))

# Q3: Please use the support vector machine(default parameters) in sklearn
# to classify the data above, and print the training accuracy and test accuracy.
# TODO:use support vector machine
clf = LinearSVC()
clf.fit(X_train, Y_train)

train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

print('Q3')
print('Training accuracy: %0.2f%%' % (train_accuracy * 100))
print('Testing accuracy: %0.2f%%' % (test_accuracy * 100))

# Q4:Please adjust the parameters of SVM to increase the testing accuracy,
# and print the training accuracy and test accuracy.
# TODO:use SVM with another group of parameters
clf = LinearSVC(C=0.01, max_iter=5000)
clf.fit(X_train, Y_train)

train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

print('Q4')
print('Training accuracy: %0.2f%%' % (train_accuracy * 100))
print('Testing accuracy: %0.2f%%' % (test_accuracy * 100))


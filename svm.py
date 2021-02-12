import numpy as np
from sklearn.svm import SVC
from sklearn import svm

from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
#files
from preprocessing import readfile, calc_length, flatten
from preprocessing import padding_data, one_hot_encoding
from preprocessing import CHANNELS
from sklearn.metrics import classification_report

folder = "data"
train_data = readfile(f"{folder}/ae.train", 0)
test_data = readfile(f"{folder}/ae.test", 1)
maxlength = calc_length(test_data, train_data)

train_input, train_output = padding_data(train_data, maxlength, 0, True)
test_input, test_output = padding_data(test_data, maxlength, 1, True)

"""
Linear classifiers SGDC
(Using default loss=hinge gives a linear SVM)
accuracy 10 folds: 0.9729
"""
cv = KFold(n_splits=10, random_state=42, shuffle=True)
sgdc = SGDClassifier(alpha = 0.03, max_iter = 1000)
scores = []
test_output = test_output.ravel()
train_output = train_output.ravel()
for train_index, test_index in cv.split(train_input):
    X_train, y_train = train_input[train_index] , train_output[train_index]
    X_test, y_test = train_input[test_index] , train_output[test_index]
    sgdc.fit(X_train, y_train)
    scores.append(sgdc.score(X_test, y_test))

test_scores = sgdc.score(test_input, test_output)
pred = sgdc.predict(test_input)
print("SGD")
print(confusion_matrix(pred, test_output))
print(test_scores)
print("\n")

"""
Linear classifiers SGDC
(Using default loss=hinge gives a linear SVM)
accuracy 10 folds: 0.9702
"""
cv = KFold(n_splits=10, random_state=42, shuffle=True)
model = SVC(kernel="rbf", gamma= 0.002, C = 41)
scores = []
for train_index, test_index in cv.split(train_input):
    X_train, y_train = train_input[train_index] , train_output[train_index]
    X_test, y_test = train_input[test_index] , train_output[test_index]
    model.fit(X_train, y_train)
    scores.append(sgdc.score(X_test, y_test))

pred = model.predict(test_input)
print("SVC")
print(confusion_matrix(pred, test_output))
print(model.score(test_input, test_output))

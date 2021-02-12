#External libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#files
from preprocessing import readfile, calc_length, flatten
from preprocessing import padding_data, one_hot_encoding
from preprocessing import CHANNELS


folder = "data"
train_data = readfile(f"{folder}/ae.train", 0)
test_data = readfile(f"{folder}/ae.test", 1)
maxlength = calc_length(test_data, train_data)

train_input, train_output = padding_data(train_data, maxlength, 0, True)
test_input, test_output = padding_data(test_data, maxlength, 1, True)

pca = PCA(n_components=CHANNELS)
train_transformed = pca.fit_transform(train_input)
test_transformed = pca.fit_transform(test_input)

"""
Logistic Regression with PCA
folds: 5, 10, 15
accuracy: 0.3162162162162162
"""
# cv = KFold(n_splits=10, random_state=42, shuffle=True)
# clf = LogisticRegression()
# scores = []
# test_output = test_output.ravel()
# train_output = train_output.ravel()
# for train_index, test_index in cv.split(train_transformed):
#     X_train, y_train = train_transformed[train_index] , train_output[train_index]
#     X_test, y_test = train_transformed[test_index] , train_output[test_index]
#     clf.fit(X_train, y_train)
#     scores.append(clf.score(X_test, y_test))

# test_scores = clf.score(test_transformed, test_output)
# pred = clf.predict(test_transformed)
# print(confusion_matrix(pred, test_output))
# print(test_scores)

"""
Logistic Regression without PCA
folds: 5, 10, 20
accuracy: 0.9675675675675676
"""
# cv = KFold(n_splits=5, random_state=42, shuffle=True)
# clf = LogisticRegression(C = 1, penalty="l2", max_iter = 1000)
# scores = []
# test_output = test_output.ravel()
# train_output = train_output.ravel()
# for train_index, test_index in cv.split(train_input):
#     X_train, y_train = train_input[train_index] , train_output[train_index]
#     X_test, y_test = train_input[test_index] , train_output[test_index]
#     clf.fit(X_train, y_train)
#     scores.append(clf.score(X_test, y_test))

# test_scores = clf.score(test_input, test_output)
# pred = clf.predict(test_input)
# print(confusion_matrix(pred, test_output))
# print(test_scores)

# cv = KFold(n_splits=5, random_state=42, shuffle=True)

"""
Logistic Regression without PCA & Cross-validation
accuracy: 0.972972972972973
"""
clf = LogisticRegression(C = 1, penalty="l2", max_iter = 1000)
scores = []
test_output = test_output.ravel()
train_output = train_output.ravel()
clf.fit(train_input, train_output)
test_scores = clf.score(test_input, test_output)
pred = clf.predict(test_input)
print(confusion_matrix(pred, test_output))
print(test_scores)
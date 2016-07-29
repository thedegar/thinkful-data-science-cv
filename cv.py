# coding=utf-8
#####################################################
# Tyler Hedegard
# 7/28/16
# Thinkful Data Science
# Cross Validation
#####################################################

from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
import numpy as np

iris = datasets.load_iris()
svc = svm.SVC(kernel='linear')

# 40% training data split
out = cross_validation.train_test_split(iris.data, test_size=0.4, random_state=5)
X_train_split = out[0]
X_test_split = out[1]
out = cross_validation.train_test_split(iris.target, test_size=0.4, random_state=5)
y_train_split = out[0]
y_test_split = out[1]

split_score = svc.score(X_train_split, y_train_split)
print("60/40 split score is {}".format(split_score)) # score = 0.9888888

# K fold cross validation
kf = cross_validation.KFold(150, n_folds=5, random_state=5)
""" <<<<<<<<<<<<< Found out how to do all this with the cross_validation method.
scores = []
for train, test in kf:
    X_train, X_test = iris.data[train], iris.data[test]
    y_train, y_test = iris.target[train], iris.target[test]
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    scores.append(score)

score = np.array(scores)
mean = score.mean()
std = score.std()
"""
cv_score = cross_validation.cross_val_score(svc, iris.data, y=iris.target, cv=kf, scoring='accuracy')
print("5 kFold scores mean = {} and std = {}".format(cv_score.mean(), cv_score.std()))

# Use F1, precision, and recall scores
f1_score = cross_validation.cross_val_score(svc, iris.data, y=iris.target, cv=kf, scoring='f1')
precision = cross_validation.cross_val_score(svc, iris.data, y=iris.target, cv=kf, scoring='precision')
recall = cross_validation.cross_val_score(svc, iris.data, y=iris.target, cv=kf, scoring='recall')

print("F1 score mean = {}...Accuracy mean = {}".format(f1_score.mean(), cv_score.mean()))
print("Accuracy is better than F1 score.")

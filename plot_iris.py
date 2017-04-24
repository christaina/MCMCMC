import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_sequential import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.4, random_state=1,
    stratify=y)

lr = LogisticRegression(
    scale=1000.0, n_jobs=-1, random_state=0, n_iter=100000)
lr.fit(X_train, y_train)
print(lr.coef_)
print("Train Accuracy")
train_acc = lr.score(X_train, y_train)
print(train_acc)

print("Test accuracy")
test_acc = lr.score(X_test, y_test)
print(test_acc)

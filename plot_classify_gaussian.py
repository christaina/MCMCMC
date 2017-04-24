from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from logistic_sequential import LogisticRegression

rng = np.random.RandomState(0)
cov = np.eye(2)*0.01
X = np.concatenate((
    rng.multivariate_normal([0, 0], np.eye(2)*0.01, size=100),
    rng.multivariate_normal([0.4, 0.4], np.eye(2)*0.01, size=100)
))
y = np.array([1]*100 + [0]*100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.4, random_state=1,
    stratify=y)

t = time()
scales = np.logspace(-3, 3, 10)
train_scores = []
test_scores = []

for scale in scales:
    lr = LogisticRegression(
        random_state=0, n_iter=100000, n_jobs=-1, scale=scale, prior_scale=0.2)
    lr.partial_fit(n_features=2)
    lr.partial_fit(X_train, y_train)
    print(lr.samples_)
    print(time() - t)
    train_score = lr.score(X_train, y_train)
    print("Train score")
    print(train_score)
    train_scores.append(train_score)
    test_score = lr.score(X_test, y_test)
    print("Test score")
    print(test_score)
    test_scores.append(test_score)

train_scores = np.array(train_scores)
test_scores = np.array(test_scores)
plt.semilogx(scales, train_scores, label="Train scores")
plt.semilogx(scales, test_scores, label="Test scores")
plt.legend(loc=0)
plt.title("Accuracy vs scale (Prior=0.2)")
plt.show()

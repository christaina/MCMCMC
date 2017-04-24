import matplotlib.pyplot as plt
import numpy as np
from logistic_sequential import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(0)
lr = LogisticRegression(random_state=0, n_iter=100000)
cov = np.eye(2)*0.01

X = np.concatenate((
    rng.multivariate_normal([0, 0], np.eye(2)*0.01, size=100),
    rng.multivariate_normal([0.4, 0.4], np.eye(2)*0.01, size=100)
))
y = np.array([1]*100 + [0]*100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, test_size=0.4, random_state=1,
    stratify=y)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("True labels")
plt.show()

lr.partial_fit(n_features=3)
lr.partial_fit(X_train, y_train)
print("Train score")
print(lr.score(X_train, y_train))
print("Test score")
print(lr.score(X_test, y_test))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.testing import assert_array_almost_equal
from mlp_sequential import MLP
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=0)

minmax = MinMaxScaler()
minmax.fit(X_train)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)

n_iter = 100
scales = np.logspace(-3, 3, 10)

train_scores = []
test_scores = []
max_acc_scores = []
for scale in scales:
    print(scale)
    mlp = MLP(
        n_hidden=8, scale=scale, n_iter=n_iter, prior_scale=0.2, random_state=0,
        local=None, init="swarm", activation="relu")
    mlp.partial_fit(n_features=4, labels=np.unique(y))
    mlp.partial_fit(X_train, y_train)

    train_pred = mlp.predict(X_train)
    train_score = accuracy_score(train_pred, y_train)
    print("Train score")
    print(train_score)
    train_scores.append(train_score)

    test_pred = mlp.predict(X_test)
    test_score = accuracy_score(test_pred, y_test)
    print("Test score")
    print(test_score)
    test_scores.append(test_score)

plt.semilogx(scales, train_scores, label="Train")
plt.semilogx(scales, test_scores, label="Test")
plt.xlabel("Scale")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

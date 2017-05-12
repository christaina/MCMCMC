import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.testing import assert_array_almost_equal
from mlp_sequential import MLP

def softmax(logits):
    logits -= np.expand_dims(np.max(logits, axis=1), axis=1)
    logits = np.exp(logits)
    logits /= np.expand_dims(np.sum(logits), axis=1)
    return logits

def softmax_1D(vec):
    vec = vec - np.max(vec)
    xform = np.exp(vec)
    xform /= np.sum(xform)
    return xform

iris = load_iris()
X, y = iris.data, iris.target
acc_scores = []
max_acc_scores = []

n_iter = 1000


mlp = MLP(
    n_hidden=8, scale=2.15, n_iter=n_iter, prior_scale=0.2,
    random_state=0, alpha=0.0,local='mh',mh_iter=100)
mlp.partial_fit(n_features=4, labels=np.unique(y))
mlp.partial_fit(X, y)

# Taking the best
s_w = mlp.rng_.multinomial(n_iter, mlp.weights_)
t = np.argmax(s_w)
wi = mlp.samples_i_[t]
wo = mlp.samples_o_[t]
probs = mlp.forward(X, wi, wo)
pred = np.argmax(probs, axis=1)
max_acc_score = accuracy_score(pred, y)
print("MAP weight score %.3f"%max_acc_score)
max_acc_scores.append(max_acc_score)

# Taking the average according to the weights
probs = mlp.predict(X)
ave_acc_score = accuracy_score(probs, y)
print("Clust weight score %.3f"%ave_acc_score)
acc_scores.append(ave_acc_score)

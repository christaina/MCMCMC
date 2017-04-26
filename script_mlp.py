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
mlp = MLP(n_hidden=8, scale=1000.0, n_iter=100000, prior_scale=0.2, random_state=0)
mlp.partial_fit(n_features=4, labels=np.unique(y))
mlp.partial_fit(X, y)

# Taking the best
s_w = mlp.rng_.multinomial(100000, mlp.weights_)
t = np.argmax(s_w)
wi = mlp.samples_i_[t]
wo = mlp.samples_o_[t]
probs = mlp.forward(X, wi, wo)
pred = np.argmax(probs, axis=1)
print(accuracy_score(pred, y))

# Taking the average according to the weights
probs = mlp.predict(X)
print(accuracy_score(probs, y))

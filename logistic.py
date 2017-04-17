import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

class LogisticRegression(LinearClassifierMixin):
    def __init__(self, scale=1.0, n_iter=20000, random_state=None):
        self.scale = scale
        self.n_iter = n_iter
        self.random_state = random_state

    def logistic_function(self, X, w):
        lin = np.matmul(X,w[:-1])+w[-1]
        return 1./ (1. + np.exp(-lin))

    def softmax(self, vec):
        xform = np.exp(vec)
        xform = xform / sum(xform)
        return xform

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        w = np.ones((X.shape[1] + 1))

        w_samples = np.ones((self.n_iter, len(w)))
        weights = np.ones(self.n_iter)
        cov = self.scale * np.eye(len(w))

        for i in range(self.n_iter):
            w_samples[i] = w
            weights[i] = -log_loss(y, self.logistic_function(X, w))
            w = rng.multivariate_normal(w, cov)

        self.weights_ = self.softmax(weights)
        self.w_samples_ = w_samples

        resamped = rng.multinomial(self.n_iter, self.weights_)
        resamp_ind = ([[i]*x for i, x in enumerate(resamped) if x > 0])
        resamp_ind = np.array([ind for inds in resamp_ind for ind in inds])
        coefs = np.mean(w_samples[resamp_ind], axis=0)
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]

    def predict(self, X):
        return np.array((np.dot(X, self.coef_) + self.intercept_ > 0), dtype=np.int)

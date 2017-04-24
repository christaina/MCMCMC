import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals.joblib import delayed, Parallel
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

def softmax(X, w):
    lin = np.dot(X, w[:-1]) + w[-1]
    return 1./ (1. + np.exp(-lin))

def _log_loss(seed, mean, cov, X, y, labels):
    rng = check_random_state(seed)
    sample = rng.multivariate_normal(mean, cov)

    n_labels = len(labels)
    if n_labels == 2:
        n_labels = 1
    sample = np.reshape(sample, (n_labels, -1))
    probs = softmax(X, sample.T)
    weight = -log_loss(y, probs, labels=labels)
    return sample, weight

class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """
    Logistic Regression using Sequential Monte Carlo.
    """
    def __init__(self, scale=1.0, n_iter=20000, random_state=None,
                 prior_scale=10.0, n_jobs=1, fit_intercept=True):
        self.scale = scale
        self.n_iter = n_iter
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.n_jobs = n_jobs
        self.fit_intercept = fit_intercept

    def softmax(self, vec):
        vec = vec - np.max(vec)
        xform = np.exp(vec)
        xform /= np.sum(xform)
        return xform

    def partial_fit(self, X=None, y=None, labels=None, n_features=None):
        # Called first time
        if X is None:
            if labels is None:
                raise ValueError("labels should be provided at first call to "
                                 "partial_fit.")
            if n_features is None:
                raise ValueError("n_features should be provided at first call "
                                 "to partial_fit.")

            self.classes_ = labels
            if self.fit_intercept:
                n_features += 1
            self.rng_ = check_random_state(self.random_state)

            if len(self.classes_) == 2:
                n_labels = 1
            total = n_features * n_labels
            self.w_ = self.rng_.multivariate_normal(
                np.zeros(total),
                self.prior_scale*np.eye(total), size=self.n_iter)
        else:
            if len(self.classes_) == 2:
                n_labels = 1
            X, y = check_X_y(X, y)

            if self.fit_intercept:
                n_features = X.shape[1] + 1
            else:
                n_features = X.shape[1]
            cov = self.scale * np.eye(self.w_.shape[-1])
            seeds = self.rng_.randint(2**32, size=self.n_iter)

            jobs = (
                delayed(_log_loss)(seed, w, cov, X, y, self.classes_)
                for seed, w in zip(seeds, self.w_)
            )
            results = np.array(Parallel(n_jobs=self.n_jobs)(jobs))
            samples = np.array([r[0] for r in results])
            weights = np.array([r[1] for r in results])
            self.samples_ = samples
            self.weights_ = self.softmax(weights)

            counts = self.rng_.multinomial(self.n_iter,self.weights_)
            w = samples[np.repeat(np.arange(self.n_iter), counts)]
            coefs = np.mean(w, axis=0)
            self.coef_ = coefs[:, : n_features-1]
            self.intercept_ = coefs[:, -1]
            self.w_ = np.reshape(w, (self.n_iter, n_labels*n_features))

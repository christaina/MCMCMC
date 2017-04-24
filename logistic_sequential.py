import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals.joblib import delayed, Parallel
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

def logistic_function(X, w):
    lin = np.dot(X, w[:-1]) + w[-1]
    return 1./ (1. + np.exp(-lin))

def _log_loss(seed, mean, cov, X, y, labels):
    rng = check_random_state(seed)
    sample = rng.multivariate_normal(mean, cov)
    log_func = logistic_function(X, sample)
    weight = -log_loss(y, log_func, labels=labels)
    return np.concatenate((sample, [weight]))

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

            self.labels_ = labels
            if self.fit_intercept:
                n_features += 1
            self.rng_ = check_random_state(self.random_state)
            self.w_ = self.rng_.multivariate_normal(
                np.zeros(n_features),
                self.prior_scale * np.eye(n_features), size=self.n_iter)
        else:
            X, y = check_X_y(X, y)

            if self.fit_intercept:
                n_features = X.shape[1] + 1
            else:
                n_features = X.shape[1]
            cov = self.scale * np.eye(n_features)
            weights = np.zeros(self.n_iter)
            seeds = self.rng_.randint(2**32, size=self.n_iter)

            jobs = (
                delayed(_log_loss)(seed, w, cov, X, y, self.labels_)
                for seed, w in zip(seeds, self.w_)
            )
            results = np.array(Parallel(n_jobs=self.n_jobs)(jobs))
            samples = results[:, :-1]
            weights = results[:, -1]

            self.samples_ = samples
            self.weights_ = self.softmax(weights)
            self.w_ = samples[np.repeat(np.arange(self.n_iter), \
                    self.rng_.multinomial(self.n_iter,self.weights_))]
        coefs = np.mean(self.w_, axis=0)
        self.coef_ = coefs[: n_features-1]
        self.intercept_ = coefs[-1]

    def predict(self, X):
        return (np.dot(X, self.coef_) + self.intercept_ > 0).astype(np.int)

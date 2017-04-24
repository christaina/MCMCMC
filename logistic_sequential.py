import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

class LogisticRegression(LinearClassifierMixin):
    """
    Logistic Regression using Sequential Monte Carlo.
    """
    def __init__(self, scale=1.0, n_iter=20000, random_state=None, prior_scale=10):
        self.scale = scale
        self.n_iter = n_iter
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.rng_ = check_random_state(random_state)

    def logistic_function(self, X, w):
        lin = np.matmul(X,w[:-1])+w[-1]
        return 1./ (1. + np.exp(-lin))

    def softmax(self, vec):
        vec = vec - np.max(vec)
        xform = np.exp(vec)
        xform /= np.sum(xform)
        return xform

    def adjust_samples(self,weight_t,w_samples_t):
        resamped = self.rng.multinomial(self.n_iter,weight_t)
        coef = np.mean(np.repeat(w_samples_t, resamped, axis=0), axis=0)
        return coef

    def partial_fit(self, X=None, y=None, labels=None, n_features=10):

        # Called first time
        if X is None:
            self.rng_ = check_random_state(self.random_state)
            self.w_ = self.rng_.multivariate_normal(
                np.zeros(n_features),
                self.prior_scale * np.eye(n_features), size=self.n_iter)
        else:
            X, y = check_X_y(X, y)
            n_features = X.shape[1] + 1
            samples = np.zeros((self.n_iter, n_features))
            cov = self.scale * np.eye(n_features)
            weights = np.zeros(self.n_iter)

            for i in range(self.n_iter):
                samples[i] = self.rng_.multivariate_normal(self.w_[i], cov)
                weights[i] = -log_loss(y, self.logistic_function(X, samples[i]),labels=labels)
            self.weights_ = self.softmax(weights)
            self.w_ = samples[np.repeat(np.arange(self.n_iter), \
                    np.random.multinomial(self.n_iter,self.weights_))]
        coefs = np.mean(self.w_, axis=0)
        self.coef_ = coefs[: n_features-1]
        self.intercept_ = coefs[-1]

    def predict(self, X):
        return (np.dot(X, self.coef_) + self.intercept_ > 0).astype(np.int)

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

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


class MLP(ClassifierMixin):
    def __init__(self, scale=1.0, n_iter=100000, mh_scale=1.0,mh_iter=10000, random_state=None,
                 prior_scale=0.2, n_hidden=10, alpha=1e-3):
        self.scale = scale
        self.mh_scale = mh_scale
        self.n_iter = n_iter
        self.mh_iter = mh_iter
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.n_hidden = n_hidden
        self.alpha = alpha

    def logistic_function(self, X, w):
        lin = np.matmul(X,w[:-1]) + w[-1]
        return 1./ (1. + np.exp(-lin))

    def forward(self, X, wi, wo):
        first = np.dot(X, wi[:-1, :]) + wi[-1]
        logits = np.dot(first, wo)
        return softmax(logits)

    def mh_step(self, X,y, wi, wo):
        mh_i = wi.flatten()
        mh_o = wo.flatten()
        prob = -log_loss(
            y, self.forward(X, wi, wo),
            labels=self.classes_) - self.alpha * (np.dot(mh_i, mh_i) + np.dot(mh_o, mh_o))

        cov_i = np.eye(len(mh_i)) * self.mh_scale
        cov_o = np.eye(len(mh_o)) * self.mh_scale
        for i in range(self.mh_iter):
            samp_i = np.random.multivariate_normal(mh_i,cov_i)
            samp_o = np.random.multivariate_normal(mh_o,cov_o)

            samp_prob = -log_loss(
                y, self.forward(X, samp_i.reshape(self.wi_.shape[1],self.n_hidden),\
                 samp_o.reshape(self.n_hidden,len(self.classes_))),
                labels=self.classes_) - \
                self.alpha * (np.dot(samp_i, samp_i) + np.dot(samp_o, samp_o))
            r = np.log(self.rng_.rand())
            if  r <= min(0, samp_prob-prob):

                mh_i = samp_i
                mh_o = samp_o
                prob = samp_prob
        return mh_i.reshape(self.wi_.shape[1],self.n_hidden), \
                    mh_o.reshape(self.n_hidden,len(self.classes_))

    def partial_fit(self, X=None, y=None, labels=None, n_features=10):
        if X is None:
            if labels is None:
                raise ValueError("labels should be provided at first call to "
                                 "partial_fit.")
            if n_features is None:
                raise ValueError("n_features should be provided at first call "
                                 "to partial_fit.")
            self.rng_ = check_random_state(self.random_state)
            n_hidden = self.n_hidden
            self.classes_ = labels

            self.wi_ = self.rng_.multivariate_normal(
                np.zeros((n_features + 1)*n_hidden),
                self.prior_scale * np.eye((n_features + 1)*n_hidden),
                size=self.n_iter)

            self.wo_ = self.rng_.multivariate_normal(
                np.zeros(n_hidden*len(labels)),
                self.prior_scale * np.eye(n_hidden*len(self.classes_)),
                size=self.n_iter)

        else:
            n_hidden = self.n_hidden
            X, y = check_X_y(X, y)
            n_features = self.wi_.shape[1] // n_hidden

            samples_i = np.zeros((self.n_iter, n_features, n_hidden))
            samples_o = np.zeros((self.n_iter, n_hidden, len(self.classes_)))
            weights = np.zeros(self.n_iter)
            cov_i = self.scale * np.eye(n_features*n_hidden)
            cov_o = self.scale * np.eye(n_hidden*len(self.classes_))

            for i in range(self.n_iter):
                s_i = self.rng_.multivariate_normal(self.wi_[i], cov_i)
                samples_i[i] = s_i.reshape(n_features, n_hidden)
                s_o = self.rng_.multivariate_normal(self.wo_[i], cov_o)
                samples_o[i] = s_o.reshape(n_hidden, len(self.classes_))

                reg = self.alpha * (np.dot(s_i, s_i) + np.dot(s_o, s_o))
                loss = -log_loss(
                    y, self.forward(X, samples_i[i], samples_o[i]),
                    labels=self.classes_)

                weights[i] = loss - reg

            self.samples_i_ = samples_i
            self.samples_o_ = samples_o
            self.weights_ = softmax_1D(weights)

            resampled = np.repeat(
                np.arange(self.n_iter),
                self.rng_.multinomial(self.n_iter, self.weights_))

            self.wi_ = self.wi_[resampled]
            self.wo_ = self.wo_[resampled]

            for i in range(self.n_iter):
                self.wi_[i], self.wo_[i] = self.mh_step(X,y,self.wi_[i],self.wo_[i])

            self.coef_i_ = np.mean(samples_i[resampled], axis=0)
            self.coef_o_ = np.mean(samples_o[resampled], axis=0)
            return self


    def predict(self, X):
        f = self.forward(X, self.coef_i_, self.coef_o_)
        return np.argmax(f, axis=1)

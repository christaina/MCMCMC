import numpy as np
from functools import partial
from swarm_optimisation import swarm_opt
from scipy.optimize import basinhopping
from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

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

def neg_log_likelihood(x, X, y, mlp):
    n_features = X.shape[1]
    wi_shape = (n_features + 1) * mlp.n_hidden
    wi, wo = x[:wi_shape], x[wi_shape:]
    reg = np.dot(wi, wi) + np.dot(wo, wo)
    wi = np.reshape(wi, ((n_features+1, mlp.n_hidden)))
    wo = np.reshape(wo, (mlp.n_hidden, len(mlp.classes_)))
    return log_loss(y, mlp.forward(X, wi, wo), labels=mlp.classes_) + mlp.alpha * reg


class MLP(ClassifierMixin):
    def __init__(self, scale=1.0, n_iter=100000, mh_scale=0.001, mh_iter=10000,
                 random_state=None, prior_scale=0.2, n_hidden=10, alpha=1e-3,
                 local="basinhopping", init='swarm'):
        self.scale = scale
        self.mh_scale = mh_scale
        self.n_iter = n_iter
        self.mh_iter = mh_iter
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.local = local
        self.init = init

    def logistic_function(self, X, w):
        lin = np.matmul(X,w[:-1]) + w[-1]
        return 1./ (1. + np.exp(-lin))

    def forward(self, X, wi, wo):
        first = np.dot(X, wi[:-1, :]) + wi[-1]
        logits = np.dot(first, wo)
        return softmax(logits)

    def mh_step(self, X, y, wi, wo):
        mh_i = wi
        mh_o = wo
        prob = -log_loss(
            y, self.forward(X, mh_i.reshape(self.samples_i_.shape[1],self.n_hidden)\
            , mh_o.reshape(self.n_hidden, len(self.classes_))),
            labels=self.classes_) - self.alpha * (np.dot(mh_i, mh_i) + np.dot(mh_o, mh_o))

        cov_i = np.eye(len(mh_i)) * self.mh_scale
        cov_o = np.eye(len(mh_o)) * self.mh_scale
        for i in range(self.mh_iter):
            samp_i = self.rng_.multivariate_normal(mh_i,cov_i)
            samp_o = self.rng_.multivariate_normal(mh_o,cov_o)

            samp_prob = -log_loss(
                y, self.forward(X, samp_i.reshape(self.samples_i_.shape[1],self.n_hidden),\
                 samp_o.reshape(self.n_hidden,len(self.classes_))),
                labels=self.classes_) - \
                self.alpha * (np.dot(samp_i, samp_i) + np.dot(samp_o, samp_o))
            r = np.log(self.rng_.rand())
            if  r <= min(0, samp_prob-prob):

                mh_i = samp_i
                mh_o = samp_o
                prob = samp_prob
        return mh_i, mh_o

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
                s_o = self.rng_.multivariate_normal(self.wo_[i], cov_o)

                if self.init == "swarm":
                    wi_len = len(self.wi_[0])
                    opt_func = partial(neg_log_likelihood, X=X, y=y, mlp=self)
                    x0 = np.concatenate((s_i, s_o))
                    x0, _ = swarm_opt(x0, opt_func)
                    s_i, s_o = x0[:wi_len], x0[wi_len:]

                samples_i[i] = s_i.reshape(n_features, n_hidden)
                samples_o[i] = s_o.reshape(n_hidden, len(self.classes_))

                reg = self.alpha * (np.dot(s_i, s_i) + np.dot(s_o, s_o))
                loss = -log_loss(
                    y, self.forward(X, samples_i[i], samples_o[i]),
                    labels=self.classes_)

                weights[i] = loss - reg

            self.samples_i_ = samples_i
            self.samples_o_ = samples_o
            self.weights_ = softmax_1D(weights)

            self.multi_ = self.rng_.multinomial(self.n_iter, self.weights_)
            resampled = np.repeat(
                np.arange(self.n_iter),
                self.multi_)

            self.wi_ = self.samples_i_[resampled].reshape(self.n_iter,-1)
            self.wo_ = self.samples_o_[resampled].reshape(self.n_iter,-1)

            if self.local not in [None, "mh", "basinhopping"]:
                raise ValueError("local should be one of None, mh or basinhopping")
            if self.local == "mh":
                for i in range(self.n_iter):
                    self.wi_[i], self.wo_[i] = self.mh_step(
                        X, y, self.wi_[i], self.wo_[i])
            elif self.local == "basinhopping":
                wi_len = len(self.wi_[0])
                for i in range(self.n_iter):
                    x0 = np.concatenate((self.wi_[i], self.wo_[i]))

                    opt_func = partial(neg_log_likelihood, X=X, y=y, mlp=self)
                    res = basinhopping(opt_func, x0)
                    self.wi_[i], self.wo_[i] = res.x[: wi_len], res.x[wi_len:]

            kmeans = KMeans(random_state=self.rng_)
            clust_w = kmeans.fit(np.concatenate([self.wi_,self.wo_],axis=1)).labels_

            best_coef_i_ = None
            best_coef_o_ = None
            max_score = None

            for k in set(clust_w):
                clust_wi_ = np.mean(
                    self.wi_[np.where(clust_w==k)],axis=0).reshape(n_features,n_hidden)
                clust_wo_ = np.mean(
                    self.wo_[np.where(clust_w==k)],axis=0).reshape(n_hidden,len(self.classes_))

                clust_score = -log_loss(y,
                        self.forward(X,clust_wi_, clust_wo_),labels=self.classes_)

                if max_score is None or clust_score > max_score:
                    best_coef_i_,best_coef_o_ = clust_wi_, clust_wo_
                    max_score = clust_score

            self.coef_i_ = best_coef_i_
            self.coef_o_ = best_coef_o_
            return self


    def predict(self, X):
        f = self.forward(X, self.coef_i_, self.coef_o_)
        return np.argmax(f, axis=1)

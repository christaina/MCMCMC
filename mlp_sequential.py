import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.metrics import log_loss

class MLP(ClassifierMixin):
    def __init__(self, scale=1.0, n_iter=20000, random_state=None, prior_scale=10):
        self.scale = scale
        self.n_iter = n_iter
        self.random_state = random_state
        self.prior_scale = prior_scale
        self.rng_ = check_random_state(random_state)

    def logistic_function(self, X, w):
        lin = np.matmul(X,w[:-1])+w[-1]
        return 1./ (1. + np.exp(-lin))

    def forward(self, X, wi, wo):
        self.ah_ = self.logistic_function(X,wi)
        self.ao_ = self.logistic_function(self.ah_,wo)
        return self.ao_[:]

    def softmax(self, vec):
        vec = vec - np.max(vec)
        xform = np.exp(vec)
        xform /= np.sum(xform)
        return xform

    def partial_fit(self,X=None,y=None,labels=[0,1],n_features=10,hidden=10):

        if X is None:
            self.rng_ = check_random_state(self.random_state)
            self.hidden_ = hidden+1
            self.classes_ = labels
            self.ai_ = np.ones((1,n_features))
            self.ah_ = np.ones((1,self.hidden_))
            self.ao_ = np.ones((1,len(self.classes_)))

            self.wi_ = self.rng_.multivariate_normal(
                np.zeros(n_features*(self.hidden_-1)),
                self.prior_scale * np.eye(n_features*(self.hidden_-1)), size=self.n_iter)

            self.wo_ = self.rng_.multivariate_normal(
                np.zeros(self.hidden_ * len(self.classes_)),
                self.prior_scale * np.eye((self.hidden_)* len(self.classes_)),\
                 size=self.n_iter)

        else:
            X, y = check_X_y(X, y)
            n_features = X.shape[1] + 1
            samples_i = np.zeros((self.n_iter, n_features, self.hidden_-1))
            samples_o = np.zeros((self.n_iter, self.hidden_, len(self.classes_)))
            cov_i = self.scale * np.eye(n_features * (self.hidden_-1))
            cov_o = self.scale * np.eye(self.hidden_ * len(self.classes_))
            weights = np.zeros(self.n_iter)

            for i in range(self.n_iter):
                samples_i[i] = self.rng_.multivariate_normal(self.wi_[i], cov_i)\
                                .reshape((n_features,self.hidden_-1))
                t = self.rng_.multivariate_normal(self.wo_[i], cov_o)\
                                .reshape((self.hidden_,len(self.classes_)))

                samples_o[i] = t
                weights[i] = -log_loss(y, self.forward(X, samples_i[i],samples_o[i]),\
                        labels=self.classes_)

            self.weights_ = self.softmax(weights)
            resampled = np.repeat(np.arange(self.n_iter), \
                    np.random.multinomial(self.n_iter,self.weights_))
            self.wi_ = samples_i[resampled]
            self.wo_ = samples_o[resampled]
        coefs_i = np.mean(self.wi_, axis=0)
        coefs_o = np.mean(self.wo_, axis=0)
        print coefs_i
        self.full_coef_ = [coefs_i,coefs_o]
        self.coef_ = [coefs_i[: n_features-1],coefs_o[: n_features-1]]
        self.intercept_ = [coefs_i[-1],coefs_o[-1]]

    def predict(self, X):
        f = self.forward(X,self.full_coef_[0],self.full_coef_[1])
        print f
        o=np.argmax((f),axis=1)

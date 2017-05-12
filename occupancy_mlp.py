import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.testing import assert_array_almost_equal
from mlp_sequential import MLP
import pandas as pd

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

data_train = pd.read_csv('/scratch/nyu/mcmc/proj/data/Occupancy-detection-data/datatraining.txt')
data_test = pd.read_csv('/scratch/nyu/mcmc/proj/data/Occupancy-detection-data/datatest.txt')

n_features = 5
feat_cols = ['Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
data = np.array(data_train[feat_cols])

X, y = data[:,:-1], data[:,-1].reshape(-1,1)
minmax = MinMaxScaler()


acc_scores = []
max_acc_scores = []
n_iter = 10

for scale in np.logspace(-3, 3, 10):
    mlp = MLP(
        n_hidden=8, scale=scale, n_iter=n_iter, prior_scale=0.2,
        random_state=0, alpha=0.0, local=None,mh_iter=0,init='swarm')
    mlp.partial_fit(n_features=n_features, labels=np.unique(y))
    preds = []
    print("Scale: %s"%scale)
    for i in range(len(X)):
        print "iter",i
        x_i = minmax.fit_transform(X[i].reshape(1,-1))
        mlp.partial_fit(-x_i, y[i])
        preds.append(mlp.predict(x_i))
        print preds[i]==y[i]
        print y[i]
    ave_acc_score = accuracy_score(preds, y)

    print ave_acc_score
    print(ave_acc_score)
    acc_scores.append(ave_acc_score)

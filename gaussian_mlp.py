import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.neural_network import MLPClassifier
from mlp_sequential import MLP
import pickle
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

data_train = pickle.load(open('gaussian_time_train.list'))
data_test = pickle.load(open('gaussian_time_test.list'))

minmax = MinMaxScaler()

acc_scores = []
max_acc_scores = []
n_iter = 10

for scale in np.logspace(-3, 3, 10):

    mlp = MLP(
        n_hidden=8, scale=scale, n_iter=n_iter, prior_scale=0.2,
        random_state=0, alpha=0.0, local=None,mh_iter=0,init='swarm')
    mlp.partial_fit(n_features=2, labels=[0,1])
    preds = []

    print("Scale: %s"%scale)
    for t,samp in enumerate(data_train):
        X,y = samp[:,:-1],samp[:,-1]
        X = minmax.fit_transform(X)

        mlp.partial_fit(X, y)
        curr_preds = mlp.predict(X)
        preds.extend(curr_preds)
        print "Batch accuracy: %s (timestep %s)"%(accuracy_score(curr_preds, y),t)

    ave_acc_score = accuracy_score(preds, data_train.reshape(-1,3)[:,-1])

    print "Final train accuracy: %s"%ave_acc_score

    X_test,y_test = data_test[0][:,:-1],data_test[0][:,-1]
    X_test = minmax.transform(X_test)
    test_preds = mlp.predict(X_test)

    print "Test accuracy %s"%accuracy_score(test_preds,y_test)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.neural_network import MLPClassifier
from mlp_sequential import MLP
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

mlp_sk = MLPClassifier(hidden_layer_sizes=(8,),alpha=1e-4)
mlp = MLP(
    n_hidden=8, scale=2.15, n_iter=n_iter, prior_scale=0.2,
    random_state=0, alpha=1e-4, local=None,mh_iter=0,init='swarm')
mlp.partial_fit(n_features=2, labels=[0,1])

sk_scores=[]
mlp_scores=[]
sk_p=[]
mlp_p=[]
for t,samp in enumerate(data_train):
    X,y = samp[:,:-1],samp[:,-1]
    X = minmax.fit_transform(X)

    mlp.partial_fit(X, y)
    mlp_sk.fit(X,y)
    curr_preds = mlp.predict(X)
    sk_preds = mlp_sk.predict(X)
    sk_score = mlp_sk.score(X,y)

    print "MLP accuracy: %s (timestep %s)"%(sk_score,t)
    print "Batch accuracy: %s (timestep %s)"%(accuracy_score(curr_preds, y),t)
    print '---'
    mlp_scores.append(accuracy_score(curr_preds, y))
    sk_scores.append(sk_score)
    mlp_p.extend(curr_preds)
    sk_p.extend(mlp_sk.predict(X))

ave_acc_score=accuracy_score(mlp_p,data_train.reshape(-1,3)[:,-1])
sk_acc_avg=accuracy_score(sk_p,data_train.reshape(-1,3)[:,-1])
print "Final train accuracy: %s"%ave_acc_score

X_test,y_test = data_test[0][:,:-1],data_test[0][:,-1]
X_test = minmax.transform(X_test)
test_preds = mlp.predict(X_test)
sk_test_preds = mlp_sk.predict(X_test)

print "Test accuracy %s"%accuracy_score(test_preds,y_test)
print "MLP Test accuracy %s"%accuracy_score(sk_test_preds,y_test)
#sk_scores.append(accuracy_score(sk_test_preds,y_test))
#mlp_scores.append(accuracy_score(test_preds,y_test))

plt.plot(np.arange(len(sk_scores)),sk_scores,c='red',label='SGD MLP',marker='o')
plt.plot(np.arange(len(mlp_scores)),mlp_scores,c='blue',label='SMC MLP',marker='o')
plt.legend()

plt.xlabel("t (timestep)")
plt.ylabel("Accuracy")
plt.title("SGD vs. SMC MLP Accuracy across timesteps")
plt.savefig('../figs/mlp_vs_sk_d50.png')
plt.show()

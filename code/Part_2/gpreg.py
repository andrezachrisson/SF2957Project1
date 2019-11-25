"""
=====================================================
Gaussian process classification (GPC)
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from keras.utils import to_categorical
# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
y = to_categorical(y)

N,d = X.shape

N = np.int(600)
Ntrain = np.int(500)
Ntest = np.int(100)


Xtrain = X[0:Ntrain-1,:]
ytrain = y[0:Ntrain-1]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]


kernel = 1.0 * RBF([1]) # isotropic kernel, 1-3.5
#kernel = DotProduct(1)
gpc_rbf = GaussianProcessRegressor(kernel=kernel).fit(Xtrain, ytrain)
yp_train = gpc_rbf.predict(Xtrain)
yp_train_one_hot = (yp_train == yp_train.max(axis=1)[:,None]).astype(int)
#print(sum(yp_train))
train_error_rate = np.mean(np.not_equal(yp_train_one_hot,ytrain))
yp_test = gpc_rbf.predict(Xtest)
yp_test_one_hot = (yp_test == yp_test.max(axis=1)[:,None]).astype(int)
test_error_rate = np.mean(np.not_equal(yp_test_one_hot,ytest))
print('Training error rate')
print(train_error_rate)
print('Test error rate')
print(test_error_rate)

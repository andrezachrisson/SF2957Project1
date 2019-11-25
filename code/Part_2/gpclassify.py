"""
=====================================================
Gaussian process classification (GPC)
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)

N,d = X.shape

N = np.int(1000)
Ntrain = np.int(900)
Ntest = np.int(100)


Xtrain = X[0:Ntrain-1,:]
ytrain = y[0:Ntrain-1]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]


kernel = 1.0 * RBF([3]) # isotropic kernel, 1-3.5
#kernel = DotProduct(1)
gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
yp_train = gpc_rbf.predict(Xtrain)
train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
yp_test = gpc_rbf.predict(Xtest)
test_error_rate = np.mean(np.not_equal(yp_test,ytest))
print('Training error rate')
print(train_error_rate)
print('Test error rate')
print(test_error_rate)

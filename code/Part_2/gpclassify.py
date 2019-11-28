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
from sklearn.gaussian_process.kernels import RationalQuadratic
import seaborn as sns
import pdb
sns.set()
# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)

N,d = X.shape
# RBF kernel
#parameter = np.array([1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
#parameter = np.array([2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6])

# Dotproduct kernel
#parameter = np.array([0.1, 0.5 ,1, 5, 10, 50, 100])

# RationalQuadratic kernel
parameter = np.array([0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4])
alpha = np.array([0.1])
print(alpha)

print(parameter)

sample_size = np.array([20, 50, 100, 500, 1000, 1500])
print(sample_size)
fig, ax = plt.subplots()

for i in parameter:
    save_error_rate_y_train =np.array([])
    save_error_rate_y_test =np.array([])

    for j in sample_size:
        N = np.int(j+100)
        Ntrain = np.int(j)
        Ntest = np.int(100)

        Xtrain = X[0:Ntrain-1,:]
        ytrain = y[0:Ntrain-1]
        Xtest = X[Ntrain:N,:]
        ytest = y[Ntrain:N]

        #kernel = 1.0 * RBF([i]) # isotropic kernel, 1-3.5
        #kernel = DotProduct(i)
        kernel = RationalQuadratic(i, alpha)
        gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
        yp_train = gpc_rbf.predict(Xtrain)
        train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
        yp_test = gpc_rbf.predict(Xtest)
        test_error_rate = np.mean(np.not_equal(yp_test,ytest))
        print('Training error rate')
        print(train_error_rate)
        save_error_rate_y_train = np.append(save_error_rate_y_train, train_error_rate)
        print('Test error rate')
        print(test_error_rate)
        save_error_rate_y_test = np.append(save_error_rate_y_test, test_error_rate)

    # Plots
    #pdb.set_trace()
    ax.plot(sample_size, save_error_rate_y_test, linestyle='--', marker='o', label=f'l ={i}')
print("")
ax.set_title(f'Test error with Rational Quadratic kernel')
ax.set_xlabel('Size of training set')
ax.set_ylabel('Error')
ax.set_ylim([0,1])
ax.legend()
fig.savefig(f'images/Size_error_RationalQuadratic_a_01.png')

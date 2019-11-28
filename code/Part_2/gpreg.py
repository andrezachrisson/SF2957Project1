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
from sklearn.gaussian_process.kernels import RationalQuadratic
from keras.utils import to_categorical
import seaborn as sns
import pdb
sns.set()
np.random.seed(1)
# import some data to play with
digits = datasets.load_digits()

X = digits.data
y = np.array(digits.target, dtype = int)
y = to_categorical(y)

N,d = X.shape

# RBF kernel
#parameter = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2])
#parameter = np.array([1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6])


# Dotproduct kernel
#parameter = np.array([0.1, 0.5 ,1, 5, 10, 50, 100])
#parameter = np.array([100])

# RationalQuadratic kernel
# RationalQuadratic kernel
parameter = np.array([0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4])
alpha = np.array([5])
print(alpha)
print(parameter)

sample_size = np.array([20, 50, 100, 500, 1000, 1500])
print(sample_size)

zero_vec_test = np.zeros(100, dtype = int)
fig, ax = plt.subplots()

for i in parameter:
    save_error_rate_y_train =np.array([])
    save_error_rate_y_test =np.array([])

    for j in sample_size:
        N = np.int(j+100)
        Ntrain = np.int(j)
        Ntest = np.int(100)

        Xtrain = X[0:Ntrain,:]
        ytrain = y[0:Ntrain]
        Xtest = X[Ntrain:N,:]
        ytest = y[Ntrain:N]

        #kernel = 1.0 * RBF([i]) # isotropic kernel, 0.2-2.6
        #kernel = DotProduct(i)
        kernel = RationalQuadratic(i, alpha)
        gpc_rbf = GaussianProcessRegressor(kernel=kernel).fit(Xtrain, ytrain)
        yp_train = gpc_rbf.predict(Xtrain)
        yp_train_one_hot = (yp_train == yp_train.max(axis=1)[:,None]).astype(int)
        #train_error_rate = np.mean(np.not_equal(yp_train_one_hot,ytrain), axis = 0)
        train_error_rate = np.mean(np.not_equal(np.sum(np.not_equal(yp_train_one_hot,ytrain),axis = 1),np.zeros(j, dtype = int)))

        yp_test = gpc_rbf.predict(Xtest)
        yp_test_one_hot = (yp_test == yp_test.max(axis=1)[:,None]).astype(int)
        #test_error_rate = np.mean(np.not_equal(yp_test_one_hot,ytest), axis = 0)
        test_error_rate = np.mean(np.not_equal(np.sum(np.not_equal(yp_test_one_hot,ytest),axis = 1),zero_vec_test))

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
fig.savefig(f'images/Size_error_reg_RationalQuadratic_a_5.png')

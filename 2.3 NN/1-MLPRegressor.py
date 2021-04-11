# Import Libraries
from sklearn.neural_network import MLPRegressor

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
# Multi-layer Perceptron regressor.

class sklearn.neural_network.MLPRegressor(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                                            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                                            random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                                            beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

=======
    - hidden_layer_sizes tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    - activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’ 
        Activation function for the hidden layer.
        - ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
        - ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        - ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
        - ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
        
    - solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’ 
        The solver for weight optimization.
        - ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
        - ‘sgd’ refers to stochastic gradient descent.
        - ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
        Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.
        
    - alpha float, default=0.0001 L2 penalty (regularization term) parameter.
    - batch_size int, default=’auto’ Size of minibatches for stochastic optimizers. 
        If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
    - learning_rate {‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
        Learning rate schedule for weight updates.

        -‘constant’ is a constant learning rate given by ‘learning_rate_init’.

        -‘invscaling’ gradually decreases the learning rate learning_rate_ at each time step ‘t’ using an inverse scaling exponent of      ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)

        -‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.

        Only used when solver=’sgd’
        
    - learning_rate_init double, default=0.001
        The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.
    - power_t double, default=0.5
        The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.
    - max_iter int, default=200
        Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
    - tol float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    - warm_start bool, default=False
        When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
    - momentum float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.
    - nesterovs_momentum bool, default=True
        Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.
    - early_stopping bool, default=False
        Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’
    - validation_fraction float, default=0.1
        The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True
    - beta_1 float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1]. Only used when solver=’adam’
    - beta_2 float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1]. Only used when solver=’adam
    - epsilonfloat, default=1e-8
        Value for numerical stability in adam. Only used when solver=’adam’
    - n_iter_no_change int, default=10
        Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’
    - max_fun int, default=15000
        Only used when solver=’lbfgs’. Maximum number of function calls. The solver iterates until convergence (determined by ‘tol’), number of iterations reaches max_iter, or this number of function calls. Note that number of function calls will be greater than or equal to the number of iterations for the MLPRegressor.
=======
hidden_layer_sizes = (20,10),
        input layers,  1 layer => 20 neurons,  2 layers => 10 neurons, output layers
=======

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape,y.shape)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print("="*25)
# ----------------------------------------------------
# Applying MLPRegressor Model 

MLPRegressorModel = MLPRegressor(activation='relu',
                                 solver='adam',
                                 learning_rate='adaptive',
                                 hidden_layer_sizes = (15),
                                 max_iter=1000,
                                 batch_size = 5,
                                 random_state=33)
# 0.8281730812610498
MLPRegressorModel.fit(X_train, y_train)
# print(MLPRegressorModel.get_params)
# print("="*10)
# ----------------------------------------------------
# Calculating Details
print('MLPRegressorModel Train Score is : ' , MLPRegressorModel.score(X_train, y_train))
print('MLPRegressorModel Test Score is : ' , MLPRegressorModel.score(X_test, y_test))
print("="*10)
# ---------------------
print("Number of outputs : ", MLPRegressorModel.n_outputs_)
print('MLPRegressorModel last activation is : ' , MLPRegressorModel.out_activation_)
print('MLPRegressorModel No. of layers is : ' , MLPRegressorModel.n_layers_)
print('MLPRegressorModel No. of iterations is : ' , MLPRegressorModel.n_iter_)
print("The number of training samples seen by the solver during fitting : ", MLPRegressorModel.t_)
print("="*10)
# ---------------------
print('MLPRegressorModel loss is : ' , MLPRegressorModel.loss_)
print("MLPRegressorModel best loss is : ", MLPRegressorModel.best_loss_) # early_stopping = False (must be)
print("MLPRegressorModel loss curve is : ", MLPRegressorModel.loss_curve_[-5:]) 
print("MLPRegressorModel loss curve length is : ", len(MLPRegressorModel.loss_curve_)) 
print("="*10)
plt.figure()
sns.lineplot(data=MLPRegressorModel.loss_curve_)
plt.show(block=False)
# ---------------------
# print(len(MLPRegressorModel.coefs_),len(MLPRegressorModel.coefs_[0]),len(MLPRegressorModel.coefs_[1]),len(MLPRegressorModel.coefs_[2]))
# print("The weight matrix corresponding to layer i : ", MLPRegressorModel.coefs_)
# print("="*10)
# ---------------------
# plt.figure(figsize=(13, 5))
# # plt.figure()
# sns.heatmap(MLPRegressorModel.coefs_[0], annot=True, fmt=".2f", yticklabels=["X_1"], annot_kws = dict(fontsize='small') )
# # plt.tight_layout()
# # plt.yticks(range(1), ["X_1"], rotation=0)
# plt.yticks(rotation=0) 
# plt.xlabel("The weight matrix corresponding to layer i")
# plt.ylabel("Input feature")
# plt.show(block=False)
# ---------------------
# # plt.figure(figsize=(5, 5))
# plt.figure()
# sns.heatmap(MLPRegressorModel.coefs_[1], annot=True, fmt=".2f" )
# # plt.tight_layout()
# plt.yticks(rotation=0) 
# plt.xlabel("The weight matrix corresponding to layer i")
# plt.ylabel("Input feature")
# plt.show(block=False)
# ---------------------
# print(len(MLPRegressorModel.intercepts_),len(MLPRegressorModel.intercepts_[0]),len(MLPRegressorModel.intercepts_[1]),len(MLPRegressorModel.intercepts_[2]))
# print(MLPRegressorModel.intercepts_)
# print("="*10)
# ---------------------
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MLPRegressorModel.predict(X_test)
print('Pred Value for MLPRegressorModel is : ' , y_pred[:5])
print('True Value for MLPRegressorModel is : ' , y_test[:5])
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=y,alpha=0.5);
sns.lineplot(x=x_axis[:,0], y=MLPRegressorModel.predict(x_axis), color='k');
# sns.lineplot(x=X[:,0], y=LinearRegressionModel.predict(X));
plt.show(block=True) 
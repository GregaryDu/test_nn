#!/usr/bin/python
# -*- coding:utf-8 

from sklearn import neural_network
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split

digits = datasets.load_digits()
x = digits.data
y = digits.target
train_x, test_x, train_y, test_y = \
	train_test_split(x, y, test_size=0.2, random_state=50)
## 1) class BernoulliRBM -- Bernoulli Restricted Boltzmann Machine(RBM)
## 2) class MLPClassifier - Multi-Layer Perceptron classifier
##		this model optimize the log-loss function
## 3) class MLPRegressor -- Multi-Layer Perceptron regressor

## === MLPClassifier === ##
'''
	---- parameters ----
	hidden_layer_sizes: tuple, default=(100,)
	activation:
				{'identity, 'logistic', 'tanh', 'relu'}
				default: 'relu'
	solver:     {''lbfgs', 'sgd', 'adam'}, default:'adam'
	alpha :     L2-penalty, default=0.0001
	batch_size :
	learning_rate: {'constant', 'invscaling', 'adaptive'}
	max_iter   : default=200
	random_state: int, default=None
	shuffle    : bool, default=True
	tol:
	learning_rate_init:	initial learning rate when learning_rate = 'constant'
	power_t    : double, default=0.001
	verbose    : bool, defalut=False
	warm_start :
	momentum   : float, default=0.9, when solver=sgd
	nesterovs_momentum: bool, default=True
	early_stopping : bool, default=False
	validation_fraction : float, default=0.1, used when early_storp=True
	heta_1:
	beta_2:	
	epsilon:  used when solver=adam
	---- attributes ----
	classes_ : class labels for each output
	loss_    : current losss computed with the loss-function
	coefs_   : list, length n_layers-1
			   ith-element is the weight matrix w.r.t layer-i
	intercepts_: list length n_layer-1
			   ith-element is the bias vec w.r.t layer-i+1
	n_iter_  : int, num of solver has ran.
	n_layer_ : int, Number of layers
	n_output_: int, Number of output
	out_activation: string
	---- method ----
	predict(self, X)
	predict_log_prob(self, X)
	predict_prob(self, X)
	partial_fit(self, X, Y): fit the model to data X-Y
	fit(self, X, Y): fit the model to data X-Y
	score(self, X, Y, sample_weight):
			return mean-accuracy ont the given data
'''

NNClassifier = neural_network.MLPClassifier(hidden_layer_sizes = (500,), \
		solver = 'sgd', alpha = 0.0, activation = 'logistic', \
		learning_rate = 'constant', learning_rate_init = 0.04, \
		max_iter = 200)
NNmodel = NNClassifier.fit(train_x, train_y)
print 'train accuracy', NNmodel.score(train_x, train_y)
print 'train confusion'
print metrics.confusion_matrix(train_y, NNmodel.predict(train_x))
print 'test accuracy', NNmodel.score(test_x, test_y)
print metrics.confusion_matrix(test_y, NNmodel.predict(test_x))
print '==== model.w ===='
print NNmodel.coefs_
print '==== model.b ===='
print NNmodel.intercepts_



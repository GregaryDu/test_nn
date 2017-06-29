#!/usr/bin/python

## @time       : 2017-04-26
## @author     : yujianmin
## @reference  : http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
## @what-to-do : try using tensorflow to make a test-softmaxLR class

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

logging.basicConfig(
        level   = logging.DEBUG,
        format  = '%(asctime)s %(filename)s[line:%(lineno)d] %(funcName)s %(levelname)s %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename= './tmp.log',
        filemode= 'w'
        )

class CSimple_test:
	def __init__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
	def __del__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
	def read_data_split(self):
		#iris = datasets.load_iris()
		digits = datasets.load_digits()
		x    = digits.data
		y    = digits.target
		#data = np.hstack((x, y.reshape((y.shape[0],1))) ## merge
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
		self.train_x = train_x
		self.train_y = train_y
		self.test_x  = test_x
		self.test_y  = test_y
	def softmax_LR_sklearn(self):
		logistic = linear_model.LogisticRegression(penalty='l2', max_iter=300, solver='newton-cg', multi_class='multinomial')
		logistic.fit(self.train_x, self.train_y)
		self.model = logistic
		logging.debug('coef_'      + str(logistic.coef_))
		logging.debug('intercept_' + str(logistic.intercept_))
		logging.debug('n_iter_'    + str(logistic.n_iter_))
		logging.debug('get_params' + str(logistic.get_params))
		pred_log_prob = logistic.predict_log_proba(self.train_x)
		print ('pred_log_prob ',type(pred_log_prob), pred_log_prob.shape)
		print (pred_log_prob[0:5, :])
		pred_prob     = logistic.predict_proba(self.train_x)
		print ('pred_prob', type(pred_prob), pred_prob.shape)
		print (pred_prob[0:5, :])
		decision_res  = logistic.decision_function(self.train_x)
		print ('decision_res', type(decision_res), decision_res.shape)
		print (decision_res[0:5, :])
		pred_res      = logistic.predict(self.train_x)
		print ('pred_res', type(pred_res), pred_res.shape)
		print (pred_res[0:5])
		accuracy      = logistic.score(self.train_x, self.train_y)
		print ('train accuracy: ', accuracy)
		accuracy      = logistic.score(self.test_x, self.test_y)
		print ('test accuracy : ', accuracy)
		## metrics index ##
		test_pred_lab = logistic.predict(self.test_x)
		print ('test accuracy : ', metrics.accuracy_score(self.test_y, test_pred_lab))
		print ('test confusion matrix :')
		print (metrics.confusion_matrix(self.test_y, test_pred_lab, np.unique(self.train_y)))

	def model_save(self):
		joblib.dump(self.model, './train_model.m')
		recall_model = joblib.load('./train_model.m')
		test_pred    = recall_model.predict(self.test_x)
		print ('reload saved model, test accuracy : ', metrics.accuracy_score(self.test_y, test_pred))
if __name__=='__main__':
	CTest = CSimple_test()
	## read the data ##
	CTest.read_data_split()
	## train a cluster-model using sklearn ##
	CTest.softmax_LR_sklearn()
	## save the model
	CTest.model_save()

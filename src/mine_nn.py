#!/usr/bin/python
# -*- coding:utf-8 -*

## @time       : 2017-06-17
## @author     : yujianmin
## @reference  : http://blog.csdn.net/yujianmin1990/article/details/49935007
## @what-to-do : try to make a 3-layer-nn by hand

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from sklearn import metrics
from sklearn import datasets
#from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

logging.basicConfig(
        level   = logging.DEBUG,
        format  = '%(asctime)s %(filename)s[line:%(lineno)d] %(funcName)s %(levelname)s %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename= './tmp.log',
        filemode= 'w'
        )

class CMyNN:
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
	def single2onehotmat(self, vec):
		# (m,) -> (m, 10), here #
		row = vec.shape[0]
		res = np.zeros((row, 10))
		for i in xrange(row):
			pos = vec[i]
			res[i, pos] = 1.0
		return res
	def read_data_split(self):
		digits = datasets.load_digits()
		x    = digits.data
		y    = digits.target
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
		self.train_x = train_x
		#self.train_y = train_y
		self.train_y = self.single2onehotmat(train_y)
		self.test_x  = test_x
		#self.test_y  = test_y
		self.test_y  = self.single2onehotmat(test_y)
		print (train_x.shape)
		print (train_y.shape)
		print (self.train_x[0:10, 0:5])
		print (self.train_y[0:10, :])
		print (train_y[0:10])
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	def delt_h(self, x):
		return -np.exp(-x)/((1+np.exp(-x))**2)
		#return self.sigmoid(x) * (1-self.sigmoid(x))
	def make_label_pred(self, pred_mat):
		return np.argmax(pred_mat, axis=1)
	def comp_mean_error(self, y, y_pred):
		return np.mean(y_pred - y, axis = 1)
	def compute_confusion(self, W_1, W_2, x, y, b_1, b_2):
		Hid_input = np.dot(x, W_1) + b_1            ## (m*64)*(64*100) --> (m*100)
		Hid_output= self.sigmoid(Hid_input)         ## (m*100)         --> (m*100)
		pred_input= np.dot(Hid_output, W_2) + b_2   ## (m*100)*(100*10)--> (m*10)
		pred      = self.sigmoid(pred_input)        ## (m*10)          --> (m*10)
		pred_label= self.make_label_pred(pred)
		real_label= self.make_label_pred(y)
		print ('accuracy : ', metrics.accuracy_score(real_label, pred_label))
		print ('confusion matrix :')
		print (metrics.confusion_matrix(real_label, pred_label, np.unique(real_label)))
		
	def my_nn(self):
		hidden_node_num = 100
		W_1 = np.reshape(np.array(np.random.normal(0, 0.003, 64*hidden_node_num)), (64, hidden_node_num))
		W_2 = np.reshape(np.array(np.random.normal(0, 0.003, 10*hidden_node_num)), (hidden_node_num, 10))
		#W_1  = np.zeros((64, hidden_node_num))
		#W_2  = np.zeros((hidden_node_num, 10))
		b_1 = np.zeros((100,))
		b_2 = np.zeros((10,))
		delta_W_1 = np.zeros((64, hidden_node_num)) ## used to save the delta-of-w1
		delta_W_2 = np.zeros((hidden_node_num, 10)) ## 
		iter_num  = 400
		alpha     = 0.0005
		for i in xrange(iter_num):
			# 1) compute predict-y
			Hid_input = np.dot(self.train_x, W_1) + b_1   ## (m*64)*(64*100) --> (m*100)
			Hid_output= self.sigmoid(Hid_input)           ## (m*100)         --> (m*100)
			Hid_output_delt = self.delt_h(Hid_output)
			pred_input= np.dot(Hid_output, W_2) + b_2     ## (m*100)*(100*10)--> (m*10)
			pred      = self.sigmoid(pred_input)          ## (m*10)          --> (m*10)
			pred_delt = self.delt_h(pred)
			#print (pred[0:10, :])#
			# 1.1) compute label-pred confusion
			print ('==== testing data ====')
			self.compute_confusion(W_1, W_2, self.test_x, self.test_y, b_1, b_2)
			print ('==== training data ====')
			self.compute_confusion(W_1, W_2, self.train_x, self.train_y, b_1, b_2)
			# 2) compute delta-each-layer
			row       = self.train_y.shape[0]
			#error     = np.mean(pred - self.train_y, 1)   ## (m*10)  --> (m*10)
			error_3   = pred - self.train_y         ## (m*10)  --> (m*10)
			delta_3   = error_3 * pred_delt         ## (m*10) .* (m*10)--> (m*10)
			delta_W_2 = np.dot(Hid_output.T, delta_3)/row   ## (m*100).T*(m*10) --> (100*10)
			error_2   = np.dot(delta_3, W_2.T)      ## --> (m*100)
			delta_2   = error_2 * Hid_output_delt   ## --> (m*100)
			delta_W_1 = np.dot(self.train_x.T, delta_2)/row ## (64*m)*(m*100)   --> (64*100)
			# 3) update  the par-w
			W_2 = W_2 - alpha*delta_W_2
			W_1 = W_1 - alpha*delta_W_1
			b_2 = b_2 - alpha*np.sum(delta_3, axis=0)/row
			b_1 = b_1 - alpha*np.sum(delta_2, axis=0)/row


		pass
	def model_save(self):
		joblib.dump(self.model, './train_model.m')
		recall_model = joblib.load('./train_model.m')
		test_pred    = recall_model.predict(self.test_x)
		print ('reload saved model, test accuracy : ', metrics.accuracy_score(self.test_y, test_pred))
if __name__=='__main__':
	CTest = CMyNN()
	CTest.read_data_split()
	CTest.my_nn()

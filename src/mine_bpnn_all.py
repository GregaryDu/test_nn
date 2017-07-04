#!/usr/bin/python
# -*- coding:utf-8 -*

## @time       : 2017-06-17
## @author     : yujianmin
## @reference  : http://blog.csdn.net/yujianmin1990/article/details/49935007
## @what-to-do : try to make a 3-layer-nn by hand (one-input-layer; one-hidden-layer; one-output-layer)

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from sklearn import metrics
from sklearn import datasets
tensorflow.examples.tutorials.mnist import input_data


class CMyNN:
	def __init__(self, hidden_nodes_list=[10, 10], batch_size=100, epoch=100):
		self.train_data = ''
		self.test_data  = ''
		self.model      = ''
		self.W          = []
		self.B          = []
		self.C          = []
	def __del__(self):
		self.train_data = ''
		self.test_data  = ''
		self.model      = ''
		self.W          = []
		self.B          = []
		self.C          = []
	def read_data(self):
		mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
		self.train_data = mnist.train
		self.test_data  = mnist.test
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	def delt_h(self, x):
		#return -np.exp(-x)/((1+np.exp(-x))**2)
		return self.sigmoid(x) * (1-self.sigmoid(x))
	def make_label_pred(self, pred_mat):
		return np.argmax(pred_mat, axis=1)
	def initial_parameters(self):
		input_node_num  = self.train_data.images.shape[1]
		output_node_num = self.train_data.labels.shape[1]
		h_layers_num    = len(self.hidden_nodes_list)
		if h_layers_num<=0: print ('please input hidden_nodes_list, such as [10, 10]'); return ''
		for i in xrange(h_layers_num + 1):
			h_node_num = hidden_node_list[i]
			h_inpu_num = hidden_node_list[i-1] if i>=1 else 0
			if i == 0:
				self.W[i] = np.reshape(np.array(np.random.normal(0, 0.001, input_node_num*h_node_num)), (input_node_num, h_node_num))
				self.C[i] = np.array(np.random.normal(0, 0.001, input_node_num)) ## 反向时的 Bias ##
				self.B[i] = np.array(np.random.normal(0, 0.001, h_node_num))     ## 正向时的 Bias ##
			elif i == len(h_layers_num):
				self.W[i] = np.reshape(np.array(np.random.normal(0, 0.001, h_node_num*output_node_num)),(h_node_num, output_node_num))
				self.C[i] = np.array(np.random.normal(0, 0.001, h_node_num))
				self.B[i] = np.array(np.random.normal(0, 0.001, output_node_num))
			else:
				self.W[i] = np.reshape(np.array(np.random.normal(0, 0.001, h_inpu_num*h_node_num)), (h_inpu_num, h_node_num))
				self.C[i] = np.array(np.random.normal(0, 0.001, h_inpu_num))
				self.B[i] = np.array(np.random.normal(0, 0.001, h_node_num))
	def my_nn(self):
		self.initial_parameters()
		middle_res = {}
		middle_res['delta_W'] = [np.zeros_like(i) for i in self.W]
		middle_res['delta_B'] = [np.zeros_like(i) for i in self.B]
		middle_res['delta_C'] = [np.zeros_like(i) for i in self.C]
		middle_res['layer_input'] = ['' for i in ]
		middle_res['layer_prob']  =

		iter_num= self.epoch*int(self.train_data.images.shape[0]/self.batch_size)
		for i in xrange(iter_num):
			batch_data = self.train_data.next_batch(self.batch_size)
			batch_x    = batch_data.images
			batch_y    = batch_data.labels
			# 1) compute predict-y
			Hid_input = np.dot(self.train_x, W_1) + B_1   ## (m*64)*(64*100) --> (m*100)
			Hid_output= self.sigmoid(Hid_input)           ## (m*100)         --> (m*100)
			#print ('Hid_output.shape', Hid_output.shape)
			Hid_output_delt = self.delt_h(Hid_output)
			pred_input= np.dot(Hid_output, W_2) + B_2     ## (m*100)*(100*10)--> (m*10)
			pred      = self.sigmoid(pred_input)          ## (m*10)          --> (m*10)
			#print ('pred.shape', pred.shape)
			pred_delt = self.delt_h(pred)
			#print (pred[0:10, :])#
			# 1.1) compute label-pred confusion
			if iter_num%100 == 0:
				#print ('==== testing data ====')
				#self.compute_confusion(W_1, W_2, self.test_x, self.test_y, B_1, B_2)
				#print ('==== training data ====')
				#self.compute_confusion(W_1, W_2, self.train_x, self.train_y, B_1, B_2)
				H2 = self.sigmoid(np.dot(self.test_x, W_1)+B_1)
				P  = self.sigmoid(np.dot(H2, W_2)+B_2)
				PL = self.make_label_pred(P)
				RL = self.make_label_pred(self.test_y)
				print ('iter_num :', i, \
						'train mean error', np.mean(pred - self.train_y), \
						'pred mean error',np.mean(P - self.test_y), \
						'pred accuracy', metrics.accuracy_score(RL, PL))
			# 2) compute delta-each-layer
			row       = self.train_y.shape[0]
			
			error_3   = pred - self.train_y         ## (m*10)  --> (m*10)
			delta_3   = error_3 * pred_delt         ## (m*10) .* (m*10)--> (m*10)
			delta_W_2 = np.dot(Hid_output.T, delta_3)   ## (m*100).T*(m*10) --> (100*10)
			
			error_2   = np.dot(delta_3, W_2.T)      ## --> (m*100)
			delta_2   = error_2 * Hid_output_delt   ## --> (m*100)
			delta_W_1 = np.dot(self.train_x.T, delta_2) ## (64*m)*(m*100)   --> (64*100)
			# 3) update  the par-w
			W_2 = W_2 - alpha * delta_W_2/row
			W_1 = W_1 - alpha * delta_W_1/row
			B_2 = B_2 - alpha * np.sum(delta_3, axis=0)/row
			B_1 = B_1 - alpha * np.sum(delta_2, axis=0)/row
#			if iter_num%5000==0:
#				print (W_2)
#				print (W_1)
#				print (B_2)
#				print (B_1)
	def my_nn_withMoment(self, iter_num, alpha, hidden_num):
		# notice: this is not a standand-moment method, just a native-moment #
		hidden_node_num = hidden_num
		input_node_num  = self.train_x.shape[1]
		output_node_num = self.train_y.shape[1]
		iter_num        = iter_num
		alpha           = alpha

		W_1 = np.reshape(np.array(np.random.normal(0, 0.001, 64*hidden_node_num)), (64, hidden_node_num))
		W_2 = np.reshape(np.array(np.random.normal(0, 0.001, 10*hidden_node_num)), (hidden_node_num, 10))
		B_1 = np.array(np.random.normal(0, 0.001, hidden_node_num))
		B_2 = np.array(np.random.normal(0, 0.001,  output_node_num))
		
		delta_W_1 = np.zeros((input_node_num, hidden_node_num)) ## used to save the delta-of-w1
		delta_W_2 = np.zeros((hidden_node_num, output_node_num)) ## 
		delta_W_1_pre = np.zeros((input_node_num, hidden_node_num)) ## used to save the delta-of-w1 before current
		delta_W_2_pre = np.zeros((hidden_node_num, output_node_num)) ## 
		sample_num    = self.train_y.shape[0]
		delta_3_pre   = np.zeros((sample_num, output_node_num))
		delta_2_pre   = np.zeros((sample_num, hidden_node_num))
		pre_alpha     = 0.8
		cur_alpha     = 1 - pre_alpha

		for i in xrange(iter_num):
			# 1) compute predict-y
			Hid_input = np.dot(self.train_x, W_1) + B_1   ## (m*64)*(64*100) --> (m*100)
			Hid_output= self.sigmoid(Hid_input)           ## (m*100)         --> (m*100)
			Hid_output_delt = self.delt_h(Hid_output)
			pred_input= np.dot(Hid_output, W_2) + B_2     ## (m*100)*(100*10)--> (m*10)
			pred      = self.sigmoid(pred_input)          ## (m*10)          --> (m*10)
			pred_delt = self.delt_h(pred)
			# 1.1) compute label-pred confusion
			if iter_num%100 == 0:
				H2 = self.sigmoid(np.dot(self.test_x, W_1)+B_1)
				P  = self.sigmoid(np.dot(H2, W_2)+B_2)
				PL = self.make_label_pred(P)
				RL = self.make_label_pred(self.test_y)
				print ('iter_num :', i, \
						'train mean error', np.mean(pred - self.train_y), \
						'pred mean error',np.mean(P - self.test_y), \
						'pred accuracy', metrics.accuracy_score(RL, PL))
			# 2) compute delta-each-layer
			row       = self.train_y.shape[0]
			
			error_3   = pred - self.train_y         ## (m*10)  --> (m*10)
			delta_3   = error_3 * pred_delt         ## (m*10) .* (m*10)--> (m*10)
			delta_W_2 = np.dot(Hid_output.T, delta_3)   ## (m*100).T*(m*10) --> (100*10)
			delta_W_2_pre = pre_alpha*delta_W_2_pre + cur_alpha*delta_W_2
			delta_3_pre   = pre_alpha*delta_3_pre + cur_alpha*delta_3

			error_2   = np.dot(delta_3, W_2.T)      ## --> (m*100)
			delta_2   = error_2 * Hid_output_delt   ## --> (m*100)
			delta_W_1 = np.dot(self.train_x.T, delta_2) ## (64*m)*(m*100)   --> (64*100)
			delta_W_1_pre = pre_alpha*delta_W_1_pre + cur_alpha*delta_W_1
			delta_2_pre   = pre_alpha*delta_2_pre + cur_alpha*delta_2

			# 3) update  the par-w
			W_2 = W_2 - alpha * delta_W_2_pre/row
			W_1 = W_1 - alpha * delta_W_1_pre/row
			B_2 = B_2 - alpha * np.sum(delta_3_pre, axis=0)/row
			B_1 = B_1 - alpha * np.sum(delta_2_pre, axis=0)/row
	def compute_confusion(self, W_1, W_2, x, y, B_1, B_2):
		Hid_input = np.dot(x, W_1) + B_1            ## (m*64)*(64*100) --> (m*100)
		Hid_output= self.sigmoid(Hid_input)         ## (m*100)         --> (m*100)
		pred_input= np.dot(Hid_output, W_2) + B_2   ## (m*100)*(100*10)--> (m*10)
		pred      = self.sigmoid(pred_input)        ## (m*10)          --> (m*10)
		pred_label= self.make_label_pred(pred)
		real_label= self.make_label_pred(y)
		print ('accuracy : ', metrics.accuracy_score(real_label, pred_label))
		print ('confusion matrix :')
		print (metrics.confusion_matrix(real_label, pred_label, np.unique(real_label)))
		print ('train_y:', self.train_y)
		print ('pred_y :', pred_label)
if __name__=='__main__':
	CTest = CMyNN()
	CTest.read_data_split()
	CTest.my_nn(1000, 10, 100)
	CTest.my_nn_withMoment(1000, 10, 100)

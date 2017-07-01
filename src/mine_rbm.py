#!/usr/bin/python
# -*- coding:utf-8 -*

## @time       : 2017-06-27
## @author     : yujianmin
## @reference  : 《Training Restricted Boltzmann Machines: An Introduction》
## @what-to-do : try to make a rbm by hand with two approximal-grad method 
##               1) k-step CD-Approximate Gradient
##               2) 1-step PCD-Approximate Gradient
##               3) k-step parallel tempering Approxiamte Gradient

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.cross_validation import train_test_split


class CMyRBM:
	def __init__(self, hidden_num = 150, iternum=1000, learningrate=0.05, k_step=1):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
		self.W = ''
		self.B = ''
		self.C = ''
		self.HNum = hidden_num
		self.k_step = k_step
		self.iternum = iternum
		self.learningrate = learningrate
	def __del__(self):
		self.train_x = ''
		self.train_y = ''
		self.test_x  = ''
		self.test_y  = ''
		self.model   = ''
		self.W = ''
		self.B = ''
		self.C = ''
		self.HNum = ''
		self.k_step = ''
		self.iternum = ''
		self.learningrate = ''
	def single2onehotmat(self, vec, span_num=1):
		'''
		# (m,) -> (m, 10*span_num)
		for example span_num == 3
			vec-> [3]
			res-> [0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0,
				   0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
		'''
		row = vec.shape[0]
		res = np.zeros((row, 10*span_num))
		for i in xrange(row):
			pos = vec[i]
			res[i, pos*span_num:(pos+1)*span_num] = 1.0
		return res
		
	def read_data_split(self):
		digits = datasets.load_digits()
		x    = digits.data
		y    = digits.target
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
		self.train_x = np.where(train_x>0, 1, 0)
		self.train_y = self.single2onehotmat(train_y, 2)
		self.test_x  = np.where(test_x>0, 1, 0)
		self.test_y  = self.single2onehotmat(test_y, 2)
		print ('train_x.shape', train_x.shape)
		print ('train_y.shape', train_y.shape)
		print ('train_x:\n', self.train_x[0:10, 0:5])
		print ('train_y:\n', self.train_y[0:10, :])
		print ('real_y:', train_y[0:10])

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	def delt_sigmoid(self, x):
		return -np.exp(-x)/((1+np.exp(-x))**2)
		#return self.sigmoid(x) * (1-self.sigmoid(x))
	def make_matrix2label(self, label_mat):
		return np.argmax(label_mat, axis=1)
	def comp_mean_error(self, y, y_pred):
		return np.mean(y_pred - y, axis = 1)
	def comp_mean_sum_error(self, y, y_pred):
		return np.mean((y_pred - y)**2)

	def compute_confusion(self, real_label, prob_label):
		print ('accuracy : ', metrics.accuracy_score(real_label, pred_label))
		print ('confusion matrix :')
		print (metrics.confusion_matrix(real_label, pred_label, np.unique(real_label)))

	def my_rbm(self):
		X = np.hstack(self.train_x, self.train_y) ## 训练X:Y的联合分布 ##
		Y = self.train_y
		input_node_num  = X.shape[1]
		print ('input_node_num:', input_node_num, \
				'x_node_num:', self.train_x.shape[1], \
				'y_node_num:', self.train_y.shape[1])
		hidden_node_num = self.HNum
		self.W = np.reshape(np.array( \
					np.random.normal(0, 0.002, input_node_num*hidden_node_num)), 
					(input_node_num, hidden_node_num))
		self.B = np.zeros((hidden_node_num))
		self.C = np.zeros((input_node_num))
		k_step = self.k_step

		for i in xrange(self.iternum):
			del_W, del_B, del_C = self.getKCDGrad(X, k=k_step)
			self.W = self.W - self.learningrate * del_W
			self.B = self.B - self.learningrate * del_B
			self.C = self.C - self.learningrate * del_C
			if i%100==0:
				h_sample = self.Sample_h_given_v(X)
				v_sample = self.Sample_v_given_h(h_sample)
				mean_error = self.comp_mean_sum_error(X, v_sample)
				print ('iter:', i, '\tmean error of training data:', mean_error)
				pred_error = self.compute_pred_error(self.train_x, self.train_y)
				print ('iter:', i, '\ttrain_predict training data:', pred_error)
				pred_error = self.compute_pred_error(self.test_x, self.test_y)
				print ('iter:', i, '\ttest_predict  testing  data:', pred_error)
			if i%500==0:
				## get the sample of some test ##
				print ('=====================================')
				input_test = self.train_x[0:10, :]
				output_test= self.train_y[0:10, :]
				self.sample_test_print(input_test, output_test)
		pass
	def compute_pred_error(self, only_x, only_y):
		x_row, x_col = only_x.shape
		V_sample     = self.get_partial_pred_by_partial_input(only_x)
		mean_error   = np.mean((only_y-V_sample)**2)
		pred_error   = self.votePredLabel(V_sample, only_y)
		return pred_error, mean_error
	def VotePredLabel(self, y_sample, only_y):
		y_sample_label = self.get_label_by_kspan(y_sample)
		y_real_label   = slef.get_label_by_kspan(only_y)
		return np.mean((y_real_label-y_sample_label)**2)
	def get_label_by_kspan(self, y_mat):
		k_span   = self.k_span
		y_row, y_col = y_mat.shape
		single_vote_res = [0 for i in range(10)]
		all_vote_res    = np.zeros((y_row,))
		for k in xrange(y_row):
			pos = np.where(y_map[k,:]==1)
			label_vote = [int(i/k_span) for i in pos]
			for i in label_vote:
				single_vote_res[i] += 1
			pred_lable = np.argmax(vote_res)
			all_vote_res[k] = pred_label
			single_vote_res = [0 for i in range(10)]
		return all_vote_res
	def get_partial_pred_by_partial_input(self, only_x)
		x_row, x_col = only_x.shape
		H_input = np.dot(only_x, self.W[0:-x_col]) + self.B
		H_prob  = self.sigmoid(H_input)
		H_sample= self.sample(H_prob)
		V_input = np.dot(H_sample, self.W[x_col:, :].T) + self.C[x_col:]
		V_prob  = self.sigmoid(V_input)
		V_sample= self.sample(V_prob)
		return V_sample
	def sample_test_print(self, only_x, only_y):
		x_row, x_col = only_x.shape
		V_sample= self.get_partial_pred_by_partial_input(only_x)
		y_row, y_col = only_y.shape
		print ('x_row:', x_row, 'x_col:', x_col, 'y_row:', y_row, 'y_col:', y_col)
		for i in xrange(y_row):
			print ('sample:', i, 'real_y:', only_y[i, :])
			print ('sample:', i, 'pred_y:', V_sample[i, :])
			
	def getKCDGrad(self, X, k=1): # del_W = np.zeros_like(W) #
		for i in range(k):
			if i == 0:
				v_input = X
			h_sample= self.Sample_h_given_v(v_input)
			v_sample= self.Sample_v_given_h(h_sample)
			v_input = v_sample
		Prob_h_v0 = self.compute_h_prob_given_v(X)
		Prob_h_vk = self.compute_h_prob_given_v(v_sample)
		## del_W ==> 原始输入的<i,j> - Kstep的<i,j>
		del_W = np.dot(X.T, Prob_h_v0) - np.dot(v_sample.T, Prob_h_vk)
		del_C = np.sum(X - v_sample, axis=0)
		del_B = np.sum(Prob_h_v0 - Prob_h_vk, axis=0)
		n_row, n_col = X.shape
		return del_W/n_row, del_B/n_row, del_C/n_row

	def Sample_h_given_v(self, v):
		h_prob  = self.compute_h_prob_given_v(v)
		h_sample= self.sample(h_prob)
		return  h_sample
	def compute_h_prob_given_v(self, v):
		h_input = np.dot(v, self.W) + self.B
		h_prob  = self.sigmoid(h_input)
		return  h_prob

	def Sample_v_given_h(self, h):
		v_prob  = self.compute_v_prob_given_h(h)
		v_sample= self.sample(v_prob)
		return  v_sample
	def compute_v_prob_given_h(self, h):
		v_input = np.dot(h, self.W.T) + self.C
		v_prob  = self.sigmoid(v_input)
		return  v_prob

	def sample(self, prob):
		if len(prob.shape)==2:
			row, col = prob.shape
			rp  = np.random.random((row, col))
		else:
			num = prob.shape[0]
			rp  = np.random.random((num))
		cha = prob - rp
		return np.where(cha>0, 1, 0)

	def k_tep_PT_approx_Grad(self):
		pass

if __name__=='__main__':
	CTest = CMyRBM(hidden_num=100, iternum=1500, learningrate=0.0009)
	CTest.read_data_split()
#CTest.my_rbm()

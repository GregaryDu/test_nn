#!/usr/bin/python
# -*- coding:utf-8 -*-


from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split

def single2onehotmat(vec):
	# (m,) -> (m, 10), here #
	row = vec.shape[0]
	res = np.zeros((row, 10))
	for i in xrange(row):
		pos = vec[i]
		res[i, pos] = 1.0
	return res
def read_data_split():
	digits = datasets.load_digits()
	x    = digits.data
	y    = digits.target
	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
	train_y = single2onehotmat(train_y)
	test_y  = single2onehotmat(test_y)
	print ('train_x.shape', train_x.shape)
	print ('test_x.shape', test_x.shape)
	print ('train_y.shape', train_y.shape)
	print ('test_y.shape', test_y.shape)
	return train_x, test_x, train_y, test_y

def tensorflow_nn(iter=1000, learning_rate=0.001, hidden_num=100):
	train_x, test_x, train_y, test_y = read_data_split()
	input_node_num  = train_x.shape[1]
	hidden_node_num = hidden_num
	output_node_num = train_y.shape[1]
	#print ('in tensorflow_nn fucntion, train_y.shape:', train_y.shape)
	#print ('output_node_num : ', output_node_num)
	iter_num        = iter
	alpha           = learning_rate

	x  = tf.placeholder("float", shape=[None, input_node_num])
	y_ = tf.placeholder("float", shape=[None, output_node_num])

	## 第一层 ##
	W_1 = tf.Variable(tf.random_normal((input_node_num, hidden_node_num), mean=0.0, stddev=0.001) )
	B_1 = tf.Variable(tf.random_normal((hidden_node_num,), mean=0.0, stddev=0.001))
	
	Hid_input = tf.matmul(x, W_1) + B_1
	Hid_output= tf.nn.sigmoid(Hid_input)

	## 第二层 ##
	W_2 = tf.Variable(tf.random_normal((hidden_node_num, output_node_num), mean=0.0, stddev=0.001) )
	B_2 = tf.Variable(tf.random_normal((output_node_num,), mean=0.0, stddev=0.001))

	pred_input = tf.matmul(Hid_output, W_2) + B_2
	pred_output= tf.nn.sigmoid(pred_input)

	## 训练 ##
	cross_entropy = -tf.reduce_sum((pred_output-y_)**2)
	#cross_entropy = -tf.reduce_sum(y_ * pred_output)
	#train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(pred_output, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	for i in xrange(iter_num):
		'''
		tmp = Hid_input.eval(feed_dict={x: train_x, y_: train_y})
		print ('Hid_input.shape', tmp.shape)
		tmp = Hid_output.eval(feed_dict={x: train_x, y_: train_y})
		print ('Hid_output.shape', tmp.shape)
		tmp = pred_input.eval(feed_dict={x: train_x, y_: train_y})
		print ('pred_input.shape', tmp.shape)
		tmp = pred_output.eval(feed_dict={x: train_x, y_: train_y})
		print ('pred_output.shape', tmp.shape)
		tmp = cross_entropy.eval(feed_dict={x: train_x, y_: train_y})
		print ('cross_entropy.shape', tmp.shape)
		'''
		train_step.run(feed_dict={x: train_x, y_: train_y})
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:train_x, y_: train_y})
			test_accuracy  = accuracy.eval(feed_dict={x:test_x,  y_: test_y})
			print (i, 'training accuracy :', train_accuracy, 'test accuracy :', test_accuracy)

	print ('test accuracy :', accuracy.eval(feed_dict={x: test_x, y_: test_y}))


if __name__=='__main__':
	tensorflow_nn(iter=1500, learning_rate=0.05, hidden_num=100)

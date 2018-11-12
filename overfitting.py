# coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# import data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# add layer for train and return output of this layer
def add_layer(intputs, in_size, out_size, active_function = None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])+0.1)
        Wx_plus_b = tf.matmul(intputs, Weights) + biases
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if active_function is None:
            outputs = Wx_plus_b
        else:
            outputs = active_function(Wx_plus_b)

        tf.summary.histogram("/outputs", outputs)
        return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, shape=[None, 64])
ys = tf.placeholder(tf.float32, shape=[None, 10])

# add layer
l1 = add_layer(xs,64,50,active_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,active_function=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar("loss",cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
# initializer all variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# add graph
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("D:\log/tarin", sess.graph)
test_writer = tf.summary.FileWriter("D:\log/test", sess.graph)

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob:1})
    if i%50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

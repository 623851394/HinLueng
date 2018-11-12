# coding:utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
# download this tarin data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

# add layer for train and return output of this layer
def add_layer(intputs, in_size, out_size, active_function = None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])+0.1)
        Wx_plus_b = tf.matmul(intputs, Weights) + biases
        if active_function is None:
            outputs=Wx_plus_b
        else:
            outputs=active_function(Wx_plus_b)
        return outputs
# define this percentage
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,shape=[None,784])
ys = tf.placeholder(tf.float32,shape=[None,10])

# add layer
prediction = add_layer(xs, 784, 10, active_function=tf.nn.softmax)

#the error between prediction and the real
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

# train_step
train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
# train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())
for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))








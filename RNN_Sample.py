#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a sample RNN code.
Using this original code, you could already get a 0.63 accuracy.
Please make some changes instead of submitting the original code.
If you have some questions about the code or data,
please contact M.M. Kuang(kuangmeng@hku.hk) or Y.Q. Deng(yqdeng@cs.hku.hk)
"""

import numpy as np
import tensorflow as tf

# Prepare Data(Training and Testing)
filename = "data.txt"
a_list = []
b_list = []
c_list = []
def str_2_list(data_list):
    ret_list = []
    for i in range(len(data_list)):
        tmp_list = data_list[i].strip().split(" ")
        tmp_ret_list = [int(tmp_list[0][1]),int(tmp_list[1]),int(tmp_list[2]),int(tmp_list[3]),int(tmp_list[4]),int(tmp_list[5]),int(tmp_list[6]),int(tmp_list[7][0])]
        ret_list.append(tmp_ret_list)
    return ret_list
with open(filename, "r") as file:
    filein = file.read().splitlines()
    for item in filein:
        tmp_list = item.strip().split(",")
        a_list.append(tmp_list[0])
        b_list.append(tmp_list[1])
        c_list.append(tmp_list[2])
a_list = str_2_list(a_list)
b_list = str_2_list(b_list)
c_list = str_2_list(c_list)

# Define the dataflow graph
time_steps = 8        # time steps which is the same as the length of the bit-string
input_dim = 2         # number of units in the input layer
hidden_dim = 10       # number of units in the hidden layer
output_dim = 1        # number of units in the output layer
binary_dim = 8
largest_number = pow(2, binary_dim)

tf.reset_default_graph()
# input X and target ouput Y
X = tf.placeholder(tf.float32, [None, time_steps, input_dim], name='x')
Y = tf.placeholder(tf.float32, [None, time_steps], name='y')

# define the RNN cell: can be simple cell, LSTM or GRU
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.sigmoid)

# values is a tensor of shape [batch_size, time_steps, hidden_dim]
# last_state is a tensor of shape [batch_size, hidden_dim]
values, last_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
values = tf.reshape(values,[time_steps, hidden_dim])

# put the values from the RNN through fully-connected layer
W = tf.Variable(tf.random_uniform([hidden_dim, output_dim], minval=-1.0,maxval=1.0), name='W')
b = tf.Variable(tf.zeros([1, output_dim]), name='b')
h = tf.nn.sigmoid(tf.matmul(values,W) + b, name='h')

# minimize loss, using ADAM as weight update rule
h_ = tf.reshape(h, [time_steps])
Y_ = tf.reshape(Y, [time_steps])
loss = tf.reduce_sum(-Y_ * tf.log(h_) - (1-Y_) * tf.log(1-h_), name='loss')
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# Launch the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Number of data for training and testing
#Please remember the total number is 5000
num4train = 4000
num4test = 1000

# train
for i in range(10):
    for j in range(num4train):
        if j % 10 == 0:
            print("Epoch: %d Example: %d is running..." % (i,j))
        a = np.array([a_list[j]], dtype=np.uint8)
        b = np.array([b_list[j]], dtype=np.uint8)
        c = np.array([c_list[j]], dtype=np.uint8)
        ab = np.c_[a,b]
        x = np.array(ab).reshape([1, binary_dim, 2])
        y = np.array(c).reshape([1, binary_dim])
        sess.run(train_step, {X: x, Y: y})

remain_result = []

#Test
for i in range(num4train + 1, num4train + num4test):
    a = a_list[i]
    b = b_list[i]
    c = c_list[i]
    ab = np.c_[a,b]
    x = np.array(ab).reshape([1, binary_dim, 2])
    y = np.array(c).reshape([1, binary_dim])

    # get predicted value
    [_probs, _loss] = sess.run([h, loss], {X: x, Y: y})
    probs = np.array(_probs).reshape([8])
    prediction = np.array([1 if p >= 0.5 else 0 for p in probs]).reshape([8])
    # Save the result
    remain_result.append([prediction, y[0]])

    # calculate error
    error = np.sum(np.absolute(y - probs))

    #print the prediction, the right y and the error.
    print("---------------")
    print(prediction)
    print(y[0])
    print(error)
    print("---------------")
    print()

sess.close()

# Get the total accuracy (Please don't change this part)
accuracy = 0
for i in range(len(remain_result)):
    len_ = len(remain_result[i][0])
    tmp_num = 0
    for j in range(len_):
        if remain_result[i][0][j] == remain_result[i][1][j]:
            tmp_num += 1
    accuracy += tmp_num / len_

accuracy /= len(remain_result)

print("Accuracy: %.4f"%(accuracy))

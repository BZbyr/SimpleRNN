#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a sample RNN code.
Please make some changes instead of submitting the original code.
If you have some questions about the code or data,
please contact Y.Q. Deng(yqdeng@cs.hku.hk) or M.M. Kuang(kuangmeng@hku.hk)
"""
#π%
import timeit as time
import numpy as np
import tensorflow as tf

# Prepare Data(Training and Testing)
filename = "data.txt"
a_list0 = []
b_list0 = []
c_list0 = []
a_list = []
b_list = []
c_list = []
def str_2_list(data_list):
    ret_list = []
    for i in range(len(data_list)):
        tmp_list = data_list[i].strip().split(" ")
        tmp_ret_list = [int(tmp_list[7][0]),int(tmp_list[6]),int(tmp_list[5]),int(tmp_list[4]),int(tmp_list[3]),int(tmp_list[2]),int(tmp_list[1]),int(tmp_list[0][1])]
        ret_list.append(tmp_ret_list)
    return ret_list

with open(filename, "r") as file:
    filein = file.read().splitlines()
    for item in filein:
        tmp_list = item.strip().split(",")
        a_list0.append(tmp_list[0])
        b_list0.append(tmp_list[1])
        c_list0.append(tmp_list[2])
a_list0 = str_2_list(a_list0)
b_list0 = str_2_list(b_list0)
c_list0 = str_2_list(c_list0)

""" index = [i for i in range(len(a_list0))]
np.random.shuffle(index)
for n in range(len(a_list0)):
    a_list.append(a_list0[index[n]])
    b_list.append(b_list0[index[n]])
    c_list.append(c_list0[index[n]]) """

print("-------")
print("finish dataset")

start = time.default_timer()
# Define the dataflow graph
time_steps = 8        # time steps which is the same as the length of the bit-string
input_dim = 2         # number of units in the input layer
hidden_dim = 4       # number of units in the hidden layer
output_dim = 1        # number of units in the output layer
binary_dim = 8
largest_number = pow(2, binary_dim)

tf.reset_default_graph()
# input X and target ouput Y
X = tf.placeholder(tf.float32, [None, time_steps, input_dim], name='x')
Y = tf.placeholder(tf.float32, [None, time_steps], name='y')

# define the RNN cell: can be simple cell, LSTM or GRU
#cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.sigmoid)
cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)


print("-----")
print(cell.state_size)
print("test")
print("-------")
# cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True)

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

#loss function、交叉熵
loss = tf.reduce_sum(-Y_ * tf.log(h_+0.00000001) - (1-Y_) * tf.log(1-h_+0.0000001), name='loss')

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.01 #初始学习率


learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=10,decay_rate=0.8)

print("GOptimizer")
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)                         
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# Launch the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Number of data for training and testing
#Please remember the total number is 5000
num4train = 4000
num4test = 1000
epoch = 10

size = 10
m = 0
a_batch_list = []
b_batch_list = []
c_batch_list = []
# train
for i in range(epoch):
    index = [i for i in range(len(a_list0))]
    np.random.shuffle(index)
    a_list.clear()
    b_list.clear()
    c_list.clear()
    for n in range(len(a_list0)):
        a_list.append(a_list0[index[n]])
        b_list.append(b_list0[index[n]])
        c_list.append(c_list0[index[n]])
    for j in range(num4train):
        if(m < size):
             a_batch_list.append(a_list[j])
             b_batch_list.append(b_list[j])
             c_batch_list.append(c_list[j])
             m = m+1
             print(m)
        
        if(m == size):
            a = np.array(a_batch_list, dtype=np.uint8) #changed
            b = np.array(b_batch_list, dtype=np.uint8) #changed
            c = np.array(c_batch_list, dtype=np.uint8) #changed
            ab = np.c_[a,b]
            x = np.array(ab).reshape([size, binary_dim, 2])
            y = np.array(c).reshape([size, binary_dim])
            sess.run(train_step, {X: x, Y: y})  #输入 x 格式化成 tf 中 X 的格式，输入然后根据 loss
            m = 0

remain_result = []

#Test
print("test")
for i in range(num4train + 1, num4train + num4test):
    a = np.array(a_list[i], dtype=np.uint8) #changed
    b = np.array(b_list[i], dtype=np.uint8) #changed
    c = np.array(c_list[i], dtype=np.uint8) #changed
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
    #print("---------------")
    #print(prediction)
    #print(y[0])
    #print(error)
    #print("---------------")
    #print()

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

elapsed = (time.default_timer()-start)
print("Time used:", elapsed)
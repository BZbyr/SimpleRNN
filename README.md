
## 1. Nan Value Trap in Loss Function Computation

The first time I ran the code, I found a lot of nan values. By analyzing the code, I found that the loss function uses cross entropy and needs to calculate the logarithm. And there is zero input in the data set. However, we can't calculate the logarithm of zero.
```
loss = tf.reduce_sum(-Y_ * tf.log(h_) - (1-Y_) * tf.log(1-h_), name='loss')
```
This is actually a horrible way of computing the cross-entropy. In some samples, `h_` and `1-h_` may be `0`. That's normally not a problem since you're not interested in those, but in the way cross_entropy is written there, it yields 0*log(0) for that particular sample. Hence the NaN. Replacing it with
```
loss = tf.reduce_sum(-Y_ * tf.log(h_+1e-8) - (1-Y_) * tf.log(1-h_+1e-8), name='loss')
```
In this way, the minimum value of `h_` and `1-h_` is replaced by a minimum value `1e-8` so that there will be no Nan value.

## 2. Shuffle Dataset in every epoch

It's very important to shuffle training data before every epoch. This makes sure your neural network is not remembering a specific order. If the order of data within each epoch is the same, then the model may use this as a way of reducing the training error, which is a sort of overfitting.

It increases the randomness and improves the generalization performance of the network. And it avoids the extreme gradient when the weight is updated due to the regular data, and avoids the over-fitting or under-fitting of the final model. 
Usually, we use stochastic gradient descent (and improvements thereon), which means that they rely on the randomness to find a minimum. Shuffling training data makes the gradients more variable, which can help convergence because it increases the likelihood of hitting a good direction.

```python
num4train = 4000
num4test = 1000
epoch = 10
# train
for i in range(epoch):
    #shuffle
    index = [i for i in range(len(a_list0))]
    np.random.shuffle(index)
    a_list.clear()
    b_list.clear()
    c_list.clear()
    for n in range(len(a_list0)):
        a_list.append(a_list0[index[n]])
        b_list.append(b_list0[index[n]])
        c_list.append(c_list0[index[n]])
    #train
    for j in range(num4train): #batch_size=1
        a = np.array(a_list[j], dtype=np.uint8)
        b = np.array(b_list[j], dtype=np.uint8)
        c = np.array(c_list[j], dtype=np.uint8)
        ab = np.c_[a,b]
        x = np.array(ab).reshape([1, binary_dim, 2])
        y = np.array(c).reshape([1, binary_dim])
        sess.run(train_step, {X: x, Y: y})  
```
## 3. SGD vs Adam

Because the dataset is small, I don't want to use very complex algorithms. And **Adam** may not converge in some cases, or may miss the global optimal solution. Another article, "[***Improving Generalization Performance by Switching from Adam to SGD***](https://arxiv.org/abs/1712.07628)", has been experimentally validated. They tested on the CIFAR-10 dataset. **Adam** converged faster than **SGD**, but the final convergence result was not as good as **SGD**.

## 4. learning rate decay

In the Fensorflow, `tf.train.GradientDescentOptimizer` is designed to use a constant learning rate for all variables in all steps. 

When training the model, this situation is usually encountered: the loss of the training set is not reduced after it has dropped to a certain extent. For example, training loss has been fluctuating back and forth between 0.7 and 0.9 and cannot be further reduced. This can usually be achieved by appropriately reducing the learning rate.

``` python 
initial_learning_rate = 0.01

learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=10,decay_rate=0.8)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
```

## 5. LSTM

In the RNN, gradient disappearance or gradient explosion problems may occur, as well as long-distance dependence problems, and LSTM solves these problems well. And the data set is 8bit data, 1 may appear in any of the 8 bits.

``` python
cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
```
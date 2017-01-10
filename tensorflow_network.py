import tensorflow as tf
import tensorflow.contrib.losses as losses
import numpy as np
import pandas as pd

# tensorflow自带了MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

# 下载mnist数据集
mnist = input_data.read_data_sets('/tmp/', one_hot=True)


# sizes[0]输入层神经元个数，sizes[-1]输出层神经元个数，feedforward
def neural_network(data, sizes):
    nextInput = data
    for s1, s2 in zip(sizes[:-1], sizes[1:]):
        layer_w_b = {'w_': tf.Variable(tf.random_normal([s1, s2])),
                     'b_': tf.Variable(tf.random_normal([s2]))}
        # w·x+b
        layer = tf.add(tf.matmul(nextInput, layer_w_b['w_']), layer_w_b['b_'])
        nextInput = tf.nn.sigmoid(layer)  # 激活函数
    return nextInput


# 定义待训练的神经网络(feedforward)
def neural_network2(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # # 定义第三层"神经元"的权重和biases
    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    # layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_1 = tf.nn.sigmoid(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    # # layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_2 = tf.nn.sigmoid(layer_2)  # 激活函数
    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    # # layer_3 = tf.nn.relu(layer_3)  # 激活函数
    layer_3 = tf.nn.sigmoid(layer_3)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_3, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# cost_func = losses.mean_squared_error(predict, Y)
# cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))


# 使用随机梯度下降法训练神经网络
def train_neural_network(X, Y, sizes, learning_rate, epochs):
    predict = neural_network(X, sizes)
    cost_func = tf.reduce_mean(tf.pow(Y - predict, 2))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        cost_func)
    epochs = epochs
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for th, epoch in enumerate(range(epochs)):
            for i in range(int(mnist.train.num_examples / batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
        print('准确率: ', acc)
        return acc


# 数字(label)只能是0-9，神经网络使用10个出口节点就可以编码表示0-9；
#  1 -> [0,1.0,0,0,0,0,0,0,0]   one_hot表示只有一个出口节点是hot
#  2 -> [0,0.1,0,0,0,0,0,0,0]
#  5 -> [0,0,0,0,0,1.0,0,0,0]
#  /tmp是macOS的临时目录，重启系统数据丢失; Linux的临时目录也是/tmp

# 定义每个层有多少'神经元''
n_input_layer = 28 * 28  # 输入层
n_layer_1 = 500  # hide layer
n_layer_2 = 1000  # hide layer
n_layer_3 = 300  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

# n_layer_1 = 100  # hide layer
# n_layer_2 = 1000  # hide layer
# n_layer_3 = 100  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 10  # 输出层
"""
层数的选择：线性数据使用1层，非线性数据使用2册, 超级非线性使用3+册。层数／神经元过多会导致过拟合
"""
# 每次使用100条数据进行训练
batch_size = 100
X = tf.placeholder('float', [None, 28 * 28])
# [None, 28*28]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')

import time

para1 = range(20, 110, 20)
para2 = range(1, 33, 3)
pAccArr = np.zeros((len(list(para1)),len(list(para2))))
for i, nn in enumerate(para1):
    for j, rate in enumerate(para2):
        sizes = []
        sizes.append(n_input_layer)
        sizes.extend([nn])
        sizes.append(n_output_layer)
        s = time.time()
        acc = train_neural_network(X, Y, sizes, rate, 15)
        pAccArr[i, j] = acc
        print(sizes, "time:", time.time() - s)

col_name = list(para2)
index = list(para1)
pd.DataFrame(pAccArr, index=index, columns=col_name).to_csv("pAccTable.csv")

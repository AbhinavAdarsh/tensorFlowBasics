# Digit classification using single layer Perceptron

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# For classifying 7, 0000000100 - One hot encoding
# each bit represents the digit

sess = tf.InteractiveSession()

# 2D tensors of shape - None (Any size), 784 - 28*28 dimensions - Single flatten image
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])  # 10 classes

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y_ = tf.matmul(x, w) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# load 100 examples
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y:batch[1]})

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy: ', accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))




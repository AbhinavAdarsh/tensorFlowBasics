import tensorflow as tf

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)

# print(node1, node2)
# sess = tf.Session()

# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# c = a*b
# sess = tf.Session()
#
# file_writer = tf.summary.FileWriter('graph', sess.graph)
# # print(sess.run([node1, node2]))
#
# print(sess.run(c))
# sess.close()

# with tf.Session() as sess:
#     output = sess.run([node1, node2])
#     print(output)
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
#
# adder_node = a + b
#
# sess = tf.Session()
#
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
#
# sess.close()

# Model Parameters
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

# Inputs and Outputs
x = tf.placeholder(tf.float32)

linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss Function
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# optimize: 0.01 = Learning Rate (change or steps)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Used for global initialization
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Run the training for 1000 times
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
print(sess.run([W,b]))

# Close the session to free memory
sess.close()

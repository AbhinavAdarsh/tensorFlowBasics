import tensorflow as tf


T, F = 1., -1.
bias = 1.

train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

train_out = [
    [T],
    [F],
    [F],
    [F],
]

# Initialize weight
w = tf.Variable(tf.random_normal([3, 1]))


# Define own step function (Can use already present sign, step or sigmoid function)
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)


# Calculate weight * input (Matrix multiplication)
output = step(tf.matmul(train_in, w))
# Get the error by subtracting output from expected output
error = tf.subtract(train_out, output)
# Calculate the mean square error
mse = tf.reduce_mean(tf.square(error))

# Compute the change in weight to be applied based on error
delta = tf.matmul(train_in, error, transpose_a=True)
# Update the weight by adding the delta to w
train = tf.assign(w, tf.add(w, delta))

# Start the session for computation
sess = tf.Session()
# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Target is the desired output (err should be zero)
err, target = 1, 0
# Define number of cycles to be run
epoch, max_epoch = 0, 10

while epoch < max_epoch and err > target:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch: ', epoch, 'mse: ', err)





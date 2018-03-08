import tensorflow as tf

T, F = 1., -1.
bias = 1.

# Define training input and output for OR gate based on truth table
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

train_out = [
    [T],
    [T],
    [T],
    [F],
]

w = tf.Variable(tf.random_normal([3, 1]))


def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    out = tf.subtract(doubled, 1)
    return out


output = step(tf.matmul(train_in, w))
error = tf.subtract(train_out, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err, target = 1, 0
epoch, max_epoch = 0, 10

while epoch < max_epoch and err > target:
    err, _ = sess.run([mse, train])
    print('epoch: ', epoch, 'mse: ', err)




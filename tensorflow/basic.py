import tensorflow as tf 
import numpy as np 

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)




# W = tf.Variable([0.3], dtype=tf.float32)
# b = tf.Variable([-0.3], dtype=tf.float32)
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)

# linear_model = W*x + b
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)

# x_train = [1, 2, 3, 4]
# y_train = [0, -1, -2, -3]

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# for i in range(1000):
#     sess.run(train, { x: x_train, y: y_train})

# cur_W, cur_b, cur_loss = sess.run([W,b,loss])
# print("W: {:} b:{:} and loss:{:}".format(cur_W, cur_b,cur_loss))

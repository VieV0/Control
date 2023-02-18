import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

def create_layer(X, n, activation):
    n_dim = int(X.shape[0])
    activation_function = None
    stddev = 0.1
    if activation == "relu":
        activation_function = tf.nn.relu
        stddev = np.sqrt(2/n_dim)
    elif activation == "sigmoid":
        activation_function = tf.sigmoid
        stddev = 1/np.sqrt(n_dim)
    W = tf.Variable(tf.truncated_normal([n, n_dim], stddev=stddev))
    b = tf.Variable(tf.zeros([n, 1]))
    Z = tf.matmul(W, X) + b
    return activation_function(Z)

def run_model(learning_epoch, train_obs, train_labels, debug=False):
    sess = tf.Session()
    cost_history = np.empty(shape=[0], dtype=float)
    sess.run(tf.global_variables_initializer())
    for epoch in range(learning_epoch+1):
        for i in range(0, train_obs.shape[0]+1, 1):
            train_obs_mini = train_obs[:, i:i + 1]
            train_labels_mini = train_labels[:, i:i + 1]
            sess.run(training_step, feed_dict = {X: train_obs_mini, Y: train_labels_mini})
        cost_ = sess.run(cost, feed_dict = {X: train_obs, Y: train_labels})
        cost_history = np.append(cost_history, cost_)
        if epoch % 10 == 0 and debug:
            print(f"Эпоха {epoch}, J = {cost_}")

    return sess, cost_history



x_train = np.array([[2.5, 10.0, 0.0],
                    [1.0, 8.0, 0.0],
                    [8.0, 7.0, 7.5]])
x_train = x_train / 10.0
y_train = np.array([[0.65, 0.65, 0.85]])
x_train = x_train.transpose()
n_dim = x_train.shape[0]

n1 = 6
n2 = 6
n_outputs = 1

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])
learning_rate_initial = 1.5
decay_step = 10000
decay_rate = 0.1
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.natural_exp_decay(learning_rate_initial, global_step, decay_step, decay_rate)
hidden1 = create_layer(X, n1, "relu")
hidden2 = create_layer(hidden1, n2, "relu")
y_ = create_layer(hidden2, n_outputs, "sigmoid")
cost = - tf.reduce_mean(Y * tf.log(y_) + (1-Y) * tf.log(1-y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step = global_step)
sess, cost_history = run_model(100, x_train, y_train, debug=False)
epochs = np.arange(len(cost_history))
plt.plot(epochs, cost_history)
plt.show()
import os

import tensorflow as tf
import pandas as pd

from tensorflow.python import debug as tf_debug

input_data = pd.read_csv('data/input_data.csv', index_col=0, parse_dates=True)[2600:]
output_data = pd.read_csv('data/output_data.csv', index_col=0, parse_dates=True)[2600:]

class sample_generator():
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.inputs = input_data.copy(deep=True)
        self.labels = output_data.copy(deep=True)

    def get_sample(self, num_sample):
        if num_sample > len(self.inputs.index):
            self.inputs = self.input_data.copy(deep=True)
            self.labels = self.output_data.copy(deep=True)
        sample_inputs = self.inputs.sample(num_sample)
        sample_labels = self.labels.loc[sample_inputs.index]
        self.inputs = self.inputs.drop(sample_inputs.index)
        self.labels = self.labels.drop(sample_labels.index)

        return sample_inputs, sample_labels


def calc_inference(input_data):
    with tf.name_scope('fc1'):
        W = tf.Variable(tf.truncated_normal([1045, 1045], stddev=0.1))
        b = tf.Variable(tf.zeros([1045]))
        h1 = tf.nn.sigmoid(tf.matmul(input_data, W) + b)

    with tf.name_scope('fc2'):
        W = tf.Variable(tf.truncated_normal([1045, 1045], stddev=0.1))
        b = tf.Variable(tf.zeros([1045]))
        h2 = tf.nn.relu(tf.matmul(h1, W) + b)

    with tf.name_scope('fc3'):
        W = tf.Variable(tf.truncated_normal([1045, 1045], stddev=0.1))
        b = tf.Variable(tf.zeros([1045]))
        h3 = tf.nn.relu(tf.matmul(h2, W) + b)

    with tf.name_scope('fc4'):
        W = tf.Variable(tf.truncated_normal([1045, 1045], stddev=0.1))
        b = tf.Variable(tf.zeros([1045]))
        h4 = tf.nn.relu(tf.matmul(h3, W) + b)

    with tf.name_scope('fc5'):
        W = tf.Variable(tf.truncated_normal([1045, 206], stddev=0.1))
        b = tf.Variable(tf.zeros([206]))
        logits = tf.matmul(h4, W) + b

    return logits


def calc_loss(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    return tf.reduce_mean(cross_entropy)


def calc_training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op


def calc_accuracy(logits, labels):
    y = tf.sigmoid(logits)
    correct = tf.multiply(y, labels)
    total = tf.reduce_sum(tf.round(y))
    accuracy = tf.reduce_sum(tf.round(correct))

    return accuracy / total


def calc_total(logits):
    y = tf.sigmoid(logits)
    total = tf.reduce_sum(tf.round(y))

    return total


def main():
    sampler = sample_generator(input_data, output_data)

    X = tf.placeholder(tf.float32, [None, 1045])
    Y = tf.placeholder(tf.float32, [None, 206])

    logits = calc_inference(X)
    loss = calc_loss(logits, Y)
    train_op = calc_training(loss, 0.05)
    accuracy = calc_accuracy(logits, Y)
    total = calc_total(logits)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #builder = tf.saved_model.builder.SavedModelBuilder('model')

    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        summary_writer = tf.summary.FileWriter('log', sess.graph)
        saver.restore(sess, tf.train.latest_checkpoint('save'))
        #sess.run(init)
        
        for i in range(100000):
            inputs, labels = sampler.get_sample(50)
            _, curr_loss = sess.run([train_op, loss], feed_dict={X: inputs, Y: labels})
            if (i+1) % 100 == 0:
                curr_accuracy = accuracy.eval(feed_dict={X: input_data, Y: output_data})
                total_trade = total.eval(feed_dict={X: input_data})
                print("Step %d, training accuracy %g" % (i+1, curr_accuracy))
                print("Step %d, training loss %g" % (i+1, curr_loss))
                print("Step %d, total recommendations %g" % (i+1, total_trade))
                saver.save(sess, 'save/model.ckpt')

main()

import tensorflow as tf
import pandas as pd
import numpy as np

from sample_generator import sampleGenerator

pse_data_loc = 'data/pse_data.csv'
wb_data_loc = 'data/wb_data.csv'
labels_loc = 'data/output_data.csv'

def calc_inference(pse_data, wb_data):
    pse_data = tf.reshape(pse_data, [-1, 90, 412, 1])
    wb_data = tf.reshape(wb_data, [-1, 5, 624, 1])

    with tf.name_scope('pse_conv'):
        W = tf.Variable(tf.truncated_normal([90, 1, 1, 5], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[5]))
        pse_h = tf.nn.relu(tf.nn.conv2d(pse_data, filter=W, strides=[1, 1, 1, 1], padding='VALID') + b)

    with tf.name_scope('wb_conv'):
        W = tf.Variable(tf.truncated_normal([5, 1, 1, 3], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[3]))
        wb_h = tf.nn.relu(tf.nn.conv2d(wb_data, filter=W, strides=[1, 1, 1, 1], padding='VALID') + b)

    with tf.name_scope('fc1'):
        num_pse = 412 * 5
        num_wb = 624 * 3
        W = tf.Variable(tf.truncated_normal([num_pse + num_wb, num_pse + num_wb], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[num_pse + num_wb]))

        pse_flat = tf.reshape(pse_h, [-1, num_pse])
        wb_flat = tf.reshape(wb_h, [-1, num_wb])
        features = tf.concat([pse_flat, wb_flat], axis=1)

        h1 = tf.nn.relu(tf.matmul(features, W) + b)

    with tf.name_scope('fc2'):
        W = tf.Variable(tf.truncated_normal([num_pse + num_wb, num_pse + num_wb], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[num_pse + num_wb]))
        h2 = tf.nn.relu(tf.matmul(h1, W) + b)

    with tf.name_scope('fc3'):
        W = tf.Variable(tf.truncated_normal([num_pse + num_wb, 206], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[206]))
        logits = tf.matmul(h2, W) + b

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
    sample = sampleGenerator(pse_data_loc, wb_data_loc, labels_loc)

    X_pse = tf.placeholder(tf.float32, [None, 90, 412])
    X_wb = tf.placeholder(tf.float32, [None, 5, 624])
    Y = tf.placeholder(tf.float32, [None, 206])

    logits = calc_inference(X_pse, X_wb)
    loss = calc_loss(logits, Y)
    train_op = calc_training(loss, 0.05)
    accuracy = calc_accuracy(logits, Y)
    total = calc_total(logits)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log', sess.graph)
        saver.restore(sess, tf.train.latest_checkpoint('save'))
        #sess.run(init)
        
        for i in range(10000):
            pse_samples, wb_samples, labels_samples = sample.retrieve_sample(10)
            _, curr_loss = sess.run([train_op, loss], feed_dict={X_pse: pse_samples, X_wb: wb_samples, Y: labels_samples})
            if (i+1) % 100 == 0:
                curr_accuracy = accuracy.eval(feed_dict={X_pse: pse_samples, X_wb: wb_samples, Y: labels_samples})
                total_trade = total.eval(feed_dict={X_pse: pse_samples, X_wb: wb_samples, Y: labels_samples})
                print("Step %d, training accuracy %g" % (i+1, curr_accuracy))
                print("Step %d, training loss %g" % (i+1, curr_loss))
                print("Step %d, total recommendations %g" % (i+1, total_trade))
                saver.save(sess, 'save/model.ckpt')

if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import pandas as pd

from sample_generator import sampleGenerator

pse_src = 'data/pse_data.csv'
wb_src = 'data/wb_data.csv'
label_src = 'data/output_data.csv'

batch_sz = 5
num_period = 90
pse_sz = 412
wb_sz = 624
label_sz = 206
lstm_sz = pse_sz + wb_sz

if __name__ == '__main__':
    X = tf.placeholder(tf.float32, [batch_sz, num_period, pse_sz + wb_sz])
    Y = tf.placeholder(tf.float32, [batch_sz, num_period, label_sz])
    
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, lstm_sz])
    X_split = tf.split(XR, num_period, 0)
    
    W = tf.Variable(tf.random_normal([lstm_sz, label_sz], stddev=0.05))
    b = tf.Variable(tf.zeros([label_sz]))
    
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_sz, forget_bias=1.0, state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm, X_split, dtype=tf.float32)
    logits = [tf.matmul(output, W) + b for output in outputs]
    
    labels = tf.unstack(Y, axis=1)
    losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in zip(logits, labels)]
    cost = tf.reduce_mean(losses)
    
    train_step = tf.train.AdagradOptimizer(0.05).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sample = sampleGenerator(pse_src, wb_src, label_src)
        sess.run(init)
    
        for i in range(10000):
            for pse, wb, labels in sample.generate_batch(batch_sz, num_period):
                batch = np.concatenate([pse, wb], axis=2)
                loss, _ = sess.run([cost, train_step], feed_dict={X: batch, Y: labels})
                print(loss)


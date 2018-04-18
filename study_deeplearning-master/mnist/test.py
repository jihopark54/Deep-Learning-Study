import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mnist = input_data.read_data_sets("dataset/MNIST/", one_hot=True)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

data_X = tf.placeholder(tf.float32, [None, 784], name='input_data')
data_Z = tf.placeholder(tf.float32, [None, 64], name= 'noise_data')

gen_w_1 = tf.get_variable('gen_w_1', shape=[64, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
gen_b_1 = tf.get_variable('gen_b_1', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
gen_w_2 = tf.get_variable('gen_w_2', shape=[128, 784], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
gen_b_2 = tf.get_variable('gen_b_2', shape=[784], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

dis_w_1 = tf.get_variable('dis_w_1', shape=[784, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
dis_b_1 = tf.get_variable('dis_b_1', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
dis_w_2 = tf.get_variable('dis_w_2', shape=[128, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
dis_b_2 = tf.get_variable('dis_b_2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

def generator(noise):
    layer_1 = tf.nn.relu(tf.matmul(noise, gen_w_1) + gen_b_1)
    return tf.sigmoid(tf.matmul(layer_1, gen_w_2) + gen_b_2)

def discriminator(data):
    layer_1 = tf.nn.relu(tf.matmul(data, dis_w_1) + dis_b_1)
    return tf.matmul(layer_1, dis_w_2) + dis_b_2

real_prob = discriminator(data_X)
gen_data = generator(data_Z)
gen_prob = discriminator(gen_data)

gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_prob, labels=tf.ones_like(gen_prob, tf.float32)))
dis_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_prob, labels=tf.ones_like(real_prob))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_prob, labels=tf.zeros_like(gen_prob)))


"""
gen_cost = 0.5 * tf.reduce_mean(gen_prob - 1) ** 2
dis_cost = 0.5 * (tf.reduce_mean((real_prob -1)**2) + tf.reduce_mean(gen_prob**2 ))
"""


gen_var = [gen_w_1, gen_b_1, gen_w_2, gen_b_2]
dis_var = [dis_w_1, dis_b_1, dis_w_2, dis_b_2]
gen_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_cost, var_list=gen_var)
dis_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dis_cost, var_list=dis_var)

step = 0
i = 0
train_iters = 30000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while step < train_iters:

        for _ in range(3):
            batch_x, batch_y = mnist.train.next_batch(32)
            feed_dict = {data_X: batch_x, data_Z: sample_z(32, 64)}
            _, dis_loss = sess.run([dis_opt, dis_cost], feed_dict=feed_dict)

        feed_dict = {data_Z: sample_z(32, 64)}

        _, gen_loss = sess.run([gen_opt, gen_cost], feed_dict=feed_dict)

        step += 1
        if step % 1000 == 0:
            print(dis_loss, gen_loss)
            feed_dict = {data_Z: sample_z(16, 64)}
            samples = sess.run(gen_data, feed_dict=feed_dict)

            fig = plot(samples)
            plt.savefig('test/test_{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
from __future__ import absolute_import, division, print_function

import sys
import os
import time
import math
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import _linear

tf.flags.DEFINE_integer('M', 32, "Number of subcodes.")
tf.flags.DEFINE_integer('K', 16, "Number of vectors in each codebook.")

tf.flags.DEFINE_integer('batch_size', 32, 'Batch size for computing embeddings')
tf.flags.DEFINE_float('lr', 0.0001, "Learning rate.")
tf.flags.DEFINE_integer('max_epochs', 300, 'number of full passes through the training data')
tf.flags.DEFINE_integer('print_every', 50, 'how often to print current loss')

tf.flags.DEFINE_string('matrix', "data/glove.42B.300d.npy", "input")
tf.flags.DEFINE_string('qmats', "data/glove.42B.300d.quan.npy", "output")
tf.flags.DEFINE_integer('n_word', 50000, 'number of words to compress, 0 for all')
FLAGS = tf.flags.FLAGS


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def graph(embedding_npy, M, K):
    vocab_size = embedding_npy.shape[0]
    emb_size = embedding_npy.shape[1]
    num_centroids = 2**K
    tau = tf.placeholder_with_default(np.array(1.0, dtype='float32'), tuple()) - 0.1

    embedding = tf.constant(embedding_npy, name="embedding")
    word_input = tf.placeholder_with_default(np.array([3,4,5], dtype="int32"), shape=[None], name="word_input")
    word_lookup = tf.nn.embedding_lookup(embedding, word_input, name="word_lookup")

    A = tf.get_variable("codebook", [M * num_centroids, emb_size])

    with tf.variable_scope("h"):
        h = tf.nn.tanh(_linear(word_lookup, M*num_centroids/2, True))
    with tf.variable_scope("logits"):
        logits_lookup = _linear(h, M*num_centroids, True)
        logits_lookup = tf.log(tf.nn.softplus(logits_lookup) + 1e-8)
    logits_lookup = tf.reshape(logits_lookup, [-1, M, num_centroids], name="logits_lookup")

    D = gumbel_softmax(logits_lookup, tau, hard=False)
    D_prime = tf.reshape(D, [-1, M * num_centroids])
    y = tf.matmul(D_prime, A)

    loss = 0.5 * tf.reduce_sum((y - word_lookup)**2, axis=1)
    loss = tf.reduce_mean(loss, name="loss")

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.Variable(0.0, trainable=False, name='learning_rate')

    max_grad_norm = 0.001
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    global_norm = tf.identity(global_norm, name="global_norm")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name="train_op")

    return word_input, tau, learning_rate, train_op, loss, global_norm, D

if __name__ == '__main__':
    print("Command line:\n" + " ".join(sys.argv))

    print(FLAGS.matrix)
    matrix = np.load(FLAGS.matrix)
    if n_word > 0:
        matrix = matrix[:FLAGS.n_word]

    vocab_size = matrix.shape[0]
    emb_size = matrix.shape[1]

    tau = 1.0
    param_init = 0.01
    batch_size = min(vocab_size, FLAGS.batch_size)

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-param_init, param_init)
        with tf.variable_scope("Graph", initializer=initializer):
            word_input, tau_t, learning_rate, train_op, loss_op, global_norm_op, D = graph(matrix, FLAGS.M, FLAGS.K)

        pmax_op = tf.reduce_mean(tf.reduce_max(D, axis=2))

        tf.global_variables_initializer().run()

        best_loss = 100000
        best_counter = 0
        done = False
        quantised_matrix = np.zeros(matrix.shape)

        curr_lr = FLAGS.lr
        sess.run(tf.assign(learning_rate, curr_lr))
        word_ids = list(range(vocab_size))
        for epoch in range(FLAGS.max_epochs):
            count = 0
            start_time = time.time()
            tot_loss = 0.0
            np.random.shuffle(word_ids)
            for start_idx in range(0, vocab_size, batch_size):
                count += 1
                end_idx = start_idx + batch_size
                words = word_ids[start_idx:end_idx]
                feed_dict = {
                    word_input: words,
                    tau_t: tau,
                }
                loss, global_norm, _, pmax = sess.run([
                    loss_op,
                    global_norm_op,
                    train_op,
                    pmax_op
                ], feed_dict)

                tot_loss += loss

            tot_loss /= count
            if tot_loss < best_loss*0.99:
                best_loss = tot_loss
                best_counter = 0
            else:
                best_counter += 1
                if best_counter >= 100:
                    best_counter = 0
                    curr_lr /= 2
                    if curr_lr < 1.e-5:
                        print('learning rate too small - stopping now')
                        done = True
                    sess.run(tf.assign(learning_rate, curr_lr))

            # Print every epoch
            time_elapsed = time.time() - start_time
            bps = count / time_elapsed
            wps = len(words) * bps
            print('%6d: loss = %6.6f, bps = %.1f, wps = %.0f, lr = %0.6f, grad_norm = %6.6f, pmax = %.2f' % (
                epoch, tot_loss,
                bps, wps,
                curr_lr, global_norm, pmax))
            start_time = time.time()

            if epoch % FLAGS.print_every == 0 or done:
                graph = sess.graph
                logits_op = graph.get_tensor_by_name('Graph/logits_lookup:0')
                codebook_op = graph.get_tensor_by_name('Graph/codebook:0')
                for start_idx in range(0, vocab_size, batch_size):
                    count += 1
                    end_idx = start_idx + batch_size
                    words = np.mod(np.arange(start_idx, end_idx), vocab_size)
                    feed_dict = {
                        word_input: words,
                        tau_t: tau,
                    }
                    logits, codebook = sess.run([
                        logits_op,
                        codebook_op
                    ], feed_dict)
                    codebook = codebook.reshape((FLAGS.M, 2**FLAGS.K, emb_size))
                    # print(logits[0][0])
                    # sys.exit()

                    for i in range(batch_size):
                        w = (start_idx + i) % vocab_size
                        quantised_matrix[w].fill(0)
                        for m in range(FLAGS.M):
                            idx = np.argmax(logits[i, m], axis=-1)
                            vector = codebook[m, idx]
                            quantised_matrix[w] += vector

                print("matrix norm: {}".format(np.linalg.norm(matrix)))
                print("qmatrix norm: {}".format(np.linalg.norm(quantised_matrix)))

                diff = matrix - quantised_matrix
                print("diff norm: {}".format(np.linalg.norm(diff)))
                print("diff min: {}".format(np.min(diff)))
                print("diff max: {}".format(np.max(diff)))
                print("diff std: {}".format(np.std(diff)))
                if done:
                    break

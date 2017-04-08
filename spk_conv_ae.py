"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt
import os
from os.path import join, isdir
import time
import pickle

logdir = 'tflogs_ae/logs_{}'.format(int(time.time()))
trlogs = join(logdir, 'train')
telogs = join(logdir, 'test')
if not isdir(trlogs):
  os.makedirs(trlogs)
if not isdir(telogs):
  os.makedirs(telogs)


# %%
def autoencoder(input_shape=[None, 26],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        '''
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        '''
        x_dim = 26
        x_tensor = tf.reshape(
            x, [-1, 1, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output
        print current_input.get_shape()

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))
    tf.summary.scalar('reconstruction_cost', cost)

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


# %%
def test_timit():
    """Test the convolutional autoencder using MNIST."""
    # %%
    import matplotlib.pyplot as plt
    import timit

    # %%
    # load MNIST as before
    trf, trl, tef, tel = timit.load_feats()
    mean_feat = np.mean(trf, axis=0)
    ae = autoencoder()

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    with tf.Session() as sess:
    
        # Merge all the summaries and write them out to tflogs
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(trlogs,
                                          sess.graph)
        test_writer = tf.summary.FileWriter(telogs)

        sess.run(tf.global_variables_initializer())

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # %%
        # Fit all training data
        batch_size = 100
        n_epochs = 500
        for epoch_i in range(n_epochs):
            for batch_i in range(trf.shape[0] // batch_size):

                st = batch_i*batch_size
                en = (batch_i+1) * batch_size
                batch_feats = trf[st:] if en >= trf.shape[0] else trf[st:en]
                #batch_labels = trl[st:] if en >= trl.shape[0] else trl[st:en]

                train = np.array([feat - mean_feat for feat in batch_feats])

                sess.run(optimizer, feed_dict={ae['x']: train})


            trsummary, trcost = sess.run([merged, ae['cost']], feed_dict={ae['x']: trf})
            train_writer.add_summary(trsummary, epoch_i)

            tesummary, tecost = sess.run([merged, ae['cost']], feed_dict={ae['x']: tef})
            test_writer.add_summary(tesummary, epoch_i)
            
            print epoch_i, trcost
            print epoch_i, tecost
            #latent = sess.run(ae['z'], feed_dict={ae['x']: train})
            #print latent.shape

            # Save model after every n epochs
            n = 10
            if epoch_i % n == 0:
                # Save the variables to disk.
                save_path = saver.save(sess, "results/conv_ae_spk/model.ckpt")
                
                tr_latent_output = sess.run(ae['z'], feed_dict={ae['x']: trf})
                print tr_latent_output.shape
                train_rep = []
                for rep_i in xrange(tr_latent_output.shape[0]):
                    train_rep.append(tr_latent_output[rep_i].flatten())
                with open("results/conv_ae_spk/train.pkl", 'wb') as trnp:
                    pickle.dump((np.array(train_rep), trl), trnp)

                te_latent_output = sess.run(ae['z'], feed_dict={ae['x']: tef})
                print te_latent_output.shape
                test_rep = []
                for rep_i in xrange(te_latent_output.shape[0]):
                    test_rep.append(te_latent_output[rep_i].flatten())
                with open("results/conv_ae_spk/test.pkl", 'wb') as tstp:
                    pickle.dump((np.array(test_rep), tel), tstp)

                print("Model saved in file: %s" % save_path)


        '''
        # %%
        # Plot example reconstructions
        n_examples = 10
        test_xs, _ = mnist.test.next_batch(n_examples)
        test_xs_norm = np.array([img - mean_img for img in test_xs])
        recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
        print(recon.shape)
        fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)))
            axs[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (784,)) + mean_img,
                    (28, 28)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()
        '''

# %%
if __name__ == '__main__':
    test_timit()

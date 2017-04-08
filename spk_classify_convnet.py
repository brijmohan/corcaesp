"""Tutorial on how to build a convnet w/ modern changes, e.g.
Batch Normalization, Leaky rectifiers, and strided convolution.
"""
# %%
import tensorflow as tf
from libs.batch_norm import batch_norm, batch_norm_dense
from libs.activations import lrelu
from libs.connections import conv2d, linear

import numpy as np
import os
from os.path import join, isdir

import timit


# %% Setup input to the network and true output label.  These are
# simply placeholders which we'll fill in later.
#mnist = MNIST()

trf, trl, tef, tel = timit.load_feats(ftype='fbank_sil')
#trf, trl, tef, tel = timit.load_convae_feats()
mean_feat = np.mean(trf, axis=0)

trf = trf - mean_feat
tef = tef - mean_feat

num_classes = trl.shape[1]


x = tf.placeholder(tf.float32, [None, 26], name='x')
y = tf.placeholder(tf.float32, [None, num_classes], name='y')

# %% We add a new type of placeholder to denote when we are training.
# This will be used to change the way we compute the network during
# training/testing.
is_training = tf.placeholder(tf.bool, name='is_training')

# %% We'll convert our MNIST vector data to a 4-D tensor:
# N x W x H x C

x_tensor = tf.reshape(x, [-1, 1, 26, 1]) # FOR MFCC-26 dim
#x_tensor = tf.reshape(x, [-1, 1, 40, 1])  # FOR CONVAE

# %% We'll use a new method called  batch normalization.
# This process attempts to "reduce internal covariate shift"
# which is a fancy way of saying that it will normalize updates for each
# batch using a smoothed version of the batch mean and variance
# The original paper proposes using this before any nonlinearities

h_1 = lrelu(batch_norm(conv2d(x_tensor, 32, name='conv1', stride_h=1, k_h=1, k_w=3, pool_size=[1, 1, 2, 1], pool_stride=[1, 1, 1, 1]),
            phase_train=is_training, scope='bn1'), name='lrelu1')

h_2 = lrelu(batch_norm(conv2d(h_1, 64, name='conv2', stride_h=1, k_h=1, k_w=3, pool_size=[1, 1, 2, 1], pool_stride=[1, 1, 1, 1]),
            phase_train=is_training, scope='bn2'), name='lrelu2')

h_3 = lrelu(batch_norm(conv2d(h_2, 64, name='conv3', stride_h=1, k_h=1, k_w=3, pool_size=[1, 1, 2, 1], pool_stride=[1, 1, 1, 1]),
            phase_train=is_training, scope='bn3'), name='lrelu3')


#h_1 = lrelu((conv2d(x_tensor, 32, name='conv1', stride_h=1, k_h=1, k_w=3, pool_size=[1, 2], pool_stride=1)), name='lrelu1')

#h_2 = lrelu((conv2d(h_1, 64, name='conv2', stride_h=1, k_h=1, k_w=3, pool_size=[1, 2], pool_stride=1)), name='lrelu2')

#h_3 = lrelu((conv2d(h_2, 64, name='conv3', stride_h=1, k_h=1, k_w=3, pool_size=[1, 2], pool_stride=1)), name='lrelu3')

h_3_flat = tf.reshape(h_3, [-1, 64 * 4]) # FOR 26 dim
#h_3_flat = tf.reshape(h_3, [-1, 64 * 5])  # FOR CONVAE

h_4 = linear(h_3_flat, 2048, scope='lin1', 
  activation=lambda x: lrelu(batch_norm_dense(x, phase_train=is_training, scope='bn4'), name='lrelu5'))
h4_dropout = tf.layers.dropout(inputs=h_4, rate=0.5, training=is_training, name='dropout1')

h_5 = linear(h4_dropout, 2048, scope='lin2', 
  activation=lambda x: lrelu(batch_norm_dense(x, phase_train=is_training, scope='bn5'), name='lrelu6'))
h5_dropout = tf.layers.dropout(inputs=h_5, rate=0.5, training=is_training, name='dropout2')

h_6 = linear(h5_dropout, num_classes, scope='lin3')

y_pred = tf.nn.softmax(h_6)

# %% Define loss/eval/training functions
#cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))

tf.summary.scalar('cross_entropy', cross_entropy)

#cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

tf.summary.scalar('accuracy', accuracy)


def get_cumu_acc(test_pred, test_truth, nframes):
  corr = 0
  tot = 0
  #print test_pred.shape
  for seg_i in range(test_pred.shape[0] // nframes):
    st = seg_i * nframes
    en = (seg_i + 1) * nframes
    pred = test_pred[st:] if en >= test_pred.shape[0] else test_pred[st:en]
    gt = test_truth[st:] if en >= test_truth.shape[0] else test_truth[st:en]
    pred = np.bincount(np.argmax(pred, 1)).argmax()
    gt = np.bincount(np.argmax(gt, 1)).argmax()
    if pred == gt:
      corr += 1
    tot += 1
  #print corr, tot
  return corr/float(tot)


# %% We now create a new session to actually perform the initialization the
# variables:
with tf.Session() as sess:

  # Merge all the summaries and write them out to tflogs
  merged = tf.summary.merge_all()

  logdir = 'tflogs_fbank_sil'
  trlogs = join(logdir, 'train')
  telogs = join(logdir, 'test')
  if not isdir(trlogs):
    os.makedirs(trlogs)
  if not isdir(telogs):
    os.makedirs(telogs)

  train_writer = tf.summary.FileWriter(trlogs,
                                        sess.graph)
  test_writer = tf.summary.FileWriter(telogs)


  sess.run(tf.global_variables_initializer())

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # %% We'll train in minibatches and report accuracy:
  n_epochs = 500
  batch_size = 100
  for epoch_i in range(n_epochs):
      
      print "Shuffling..."
      ptr = np.random.permutation(len(trf))

      trf = trf[ptr]
      trl = trl[ptr]
      
      for batch_i in range(trf.shape[0] // batch_size):
          #batch_xs, batch_ys = mnist.train.next_batch(batch_size)

          st = batch_i*batch_size
          en = (batch_i+1) * batch_size
          batch_feats = trf[st:] if en >= trf.shape[0] else trf[st:en]
          batch_labels = trl[st:] if en >= trl.shape[0] else trl[st:en]

          #print batch_feats.shape, batch_labels.shape

          sess.run(train_step, feed_dict={
            x: batch_feats, y: batch_labels, is_training: True})



      trsummary, tracc = sess.run([merged, accuracy], 
                                feed_dict={
                                     x: trf,
                                     y: trl,
                                     is_training: False
                                 })
      train_writer.add_summary(trsummary, epoch_i)


      tesummary, teacc = sess.run([merged, accuracy],
                                feed_dict={
                                     x: tef,
                                     y: tel,
                                     is_training: False
                                 })
      test_writer.add_summary(tesummary, epoch_i)

      test_pred = sess.run(y_pred,
                                feed_dict={
                                     x: tef,
                                     y: tel,
                                     is_training: False
                                 })
      # Calculate accuracy for 0.5 and 1 sec
      half_sec_acc = get_cumu_acc(test_pred, tel, 50)
      one_sec_acc = get_cumu_acc(test_pred, tel, 100)

      
      print "Training accuracy..."
      print tracc

      print "Testing accuracy..."
      print teacc, half_sec_acc, one_sec_acc

      # Save model after every n epochs
      n = 10
      if epoch_i % n == 0:
        # Save the variables to disk.
        save_path = saver.save(sess, "results/fbank_sil/model.ckpt")
        print("Model saved in file: %s" % save_path)

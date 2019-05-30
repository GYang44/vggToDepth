"""
Simple tester for the vgg19_trainable
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys

from vgg19_trainable import Vgg19
import utils

import numpy as np

from tensorflow.python.framework import ops

ops.reset_default_graph()

dataSet = utils.prepareData('/media/gyang/INO1/Archive/DepthTraining/depthPhoto')
# create list of directory to color inputs and depth outputs
#dataSet = utils.prepareData('D:\\Archive\\DepthTraining\\depthPhoto')

batchSize = 10
cycles = 1000
isTraining = True

with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 224, 224, 1])
    train_mode = tf.placeholder(tf.bool)

    vgg = Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    cost = tf.losses.mean_squared_error(vgg.depth, true_out)
    cost_print_op = tf.print("cost: ", cost)

    with tf.control_dependencies([cost_print_op]):
        train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    batch, true_depth = utils.createBatch(dataSet, batchSize, [224,224])

    for i in range(0, cycles):
        #load input and output image
        sess.run(train, feed_dict={images: batch, true_out: true_depth, train_mode: isTraining})
        param = sess.run(vgg.var_dict)
        print('Cycle: ' + str(i) + " mean: " + str(np.mean(param[("conv11_2", 0)])))

    
    #test trining result
    depth = sess.run(vgg.depth, feed_dict={images: batch, train_mode: False})
    utils.show_image(depth[0].reshape(224,224))
    
    # test save
    #vgg.save_npy(sess, './test-save.npy')
    sess.close()
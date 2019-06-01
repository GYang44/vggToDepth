"""
Simple tester for the vgg19_trainable
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys

from DepthNet_MOD import Vgg19
#from vgg19_trainable import Vgg19
import utils

import numpy as np

from tensorflow.python.framework import ops

ops.reset_default_graph()

# create list of directory to color inputs and depth outputs

#dataSet = utils.DataHandler('/media/gyang/INO1/Archive/DepthTraining/depthPhoto', [224,224], 10)
dataSet = utils.DataHandler('D:\\Archive\\DepthTraining\\depthPhoto', [224,224], 10)

cycles = 10000
isTraining = True

with tf.device('/gpu:0'):
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess = tf.Session()
    
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, 224, 224, 1])
    train_mode = tf.placeholder(tf.bool)

    vgg = Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    cost = tf.losses.mean_squared_error(vgg.depth, true_out)

    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    costAvg = np.empty([cycles])
    for i in range(0, cycles):
        dataSet.reBatch()
        costAvg[i] = 0
        while dataSet.nextBatch():
            inputImage, true_depth = dataSet.getBatch()
    
            #load input and output image
            _, costValue = sess.run([train, cost], feed_dict={images: inputImage, true_out: true_depth, train_mode: isTraining})
            param = sess.run(vgg.var_dict)
            print('Cycle: {}, cost value: {}'.format(i, costValue) )
            costAvg[i] = costAvg[i] + costValue
        costAvg[i] = costAvg[i]/dataSet.getBatchCount()
    """
    #test trining result
    depth = sess.run(vgg.depth, feed_dict={images: inputImage, train_mode: False})
    for i in range(0, inputImage.shape[0]):
        utils.show_image(depth[i].reshape(224,224))
    """
    #np.save('./cost.npy', costAvg)
    # test save
    #vgg.save_npy(sess, './test-save.npy')
    sess.close()
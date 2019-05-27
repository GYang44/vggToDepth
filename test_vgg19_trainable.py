"""
Simple tester for the vgg19_trainable
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import vgg19_trainable as vgg19
import utils

# create list of directory to color inputs and depth outputs
dataSet = utils.prepareData('D:\\Archive\\DepthTraining\\depthPhoto')
#dataSet = utils.prepareData('/media/gyang/INO1/Archive/DepthTraining/depthPhoto')

with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 224, 224, 1])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    cost = tf.losses.mean_squared_error(vgg.depth, true_out)
    cost_print_op = tf.print(cost)
    with tf.control_dependencies([cost_print_op]):
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    for data in dataSet[:1]:
        #load input and output image
        color, depth = data
        inImage = utils.load_image(color)
        batch1 = inImage.reshape((1, 224, 224, 3))
        outImage = utils.load_image(depth, isGray = True)
        img1_true_result = outImage.reshape((1, 224, 224, 1))
        sess.run(train, feed_dict={images: batch1, true_out: img1_true_result, train_mode: True})
        print(' ', color, ' ', depth)
        """
        test_file = open('variableName.txt', 'w')
        for n in tf.get_default_graph().as_graph_def().node:
            test_file.write(n.name+'\n')
        """
    #test trining result
    image = utils.load_image('D:\\Archive\\DepthTraining\\depthPhoto\\Player_0_2019-04-16_12-57-29_StereoR.png')
    #image = utils.load_image('/media/gyang/INO1/Archive/DepthTraining/depthPhoto/Player_0_2019-04-16_12-57-29_StereoR.png')
    batch1 = image.reshape((1,224,224,3))
    depth = sess.run(vgg.depth, feed_dict={images: batch1, train_mode: False})
    utils.show_image(depth[0].reshape(224,224))
        
    """    
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')
    """
    # test save
    #vgg.save_npy(sess, './test-save.npy')
    sess.close()
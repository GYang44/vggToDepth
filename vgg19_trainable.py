import tensorflow as tf
import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.trainable = (train_mode is not None) & False

        self.conv1_1 = self.conv_layer(bgr, 3, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 3, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 3, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 3, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 3, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 3, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 3, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 3, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 3, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 3, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 3, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 3, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout) 

        self.trainable = (train_mode is not None) & True

        self.fc7_1 = self.fc_layer(self.fc6, 4096, 25088, "fc7_1") #12544 = 512*7*7
        #self.fc7_2 = tf.reshape(self.fc7_1, self.pool5.get_shape())
        self.tn5_7_1 = self.conv_layer(self.pool5, 3, 512, 256, "tn5_7_1")
        self.tn5_7_2 = self.conv_layer(self.tn5_7_1, 3, 256, 512, "tn5_7_2")
        self.conv7_1 = self.sum(tf.reshape(self.fc7_1, self.pool5.get_shape()), self.tn5_7_2, "conv7_1")
        self.upool7 = self.un_pool(self.conv7_1, "upool7")        
        self.conv7_2 = self.conv_layer(self.upool7, 3, 512, 512, "conv7_2")

        self.tn4_8_1 = self.conv_layer(self.pool4, 3, 512, 256, "tn4_8_1")
        self.tn4_8_2 = self.conv_layer(self.tn4_8_1, 3, 256, 512, "tn4_8_2")
        self.conv8_1 = self.sum(self.conv7_2, self.tn4_8_2, "conv8_1")
        self.upool8 = self.un_pool(self.conv8_1, "upool8")
        self.conv8_2 = self.conv_layer(self.upool8, 3, 512, 256, "conv8_2")

        self.tn3_9_1 = self.conv_layer(self.pool3, 3, 256, 128, "tn3_9_1")
        self.tn3_9_2 = self.conv_layer(self.tn3_9_1, 3, 128, 256, "tn3_9_2")
        self.conv9_1 = self.sum(self.conv8_2, self.tn3_9_2, "conv9_1")
        self.upool9 = self.un_pool(self.conv9_1, "upool9")
        self.conv9_2  = self.conv_layer(self.upool9, 3, 256, 128, "conv9_2")

        self.tn2_10_1 = self.conv_layer(self.pool2, 5, 128, 64, "tn2_10_1")
        self.tn2_10_2 = self.conv_layer(self.tn2_10_1, 5, 64, 128, "tn2_10_2")
        self.conv10_1 = self.sum(self.conv9_2, self.tn2_10_2, "conv10_1")
        self.upool10 = self.un_pool(self.conv10_1, "upool10")
        self.conv10_2 = self.conv_layer(self.upool10, 3, 128, 64, "conv10_2")

        self.tn1_11_1 = self.conv_layer(self.pool1, 5, 64, 32, "tn1_11_1")
        self.tn1_11_2 = self.conv_layer(self.tn1_11_1, 5, 32, 64, "tn1_11_2")
        self.conv11_1 = self.sum(self.conv10_2, self.tn1_11_2, "conv11_1")
        self.upool11 = self.un_pool(self.conv11_1, "upool11")
        self.conv11_2 = self.conv_layer(self.upool11, 3, 64, 3, "conv11_2")

        self.depth = self.conv_layer(self.conv11_2, 3, 3, 1, "depth")
        pass
        """
        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        """

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def un_pool(self, value, name):
        with tf.variable_scope(name):
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=name)
        return out

    def conv_layer(self, bottom, filter_size, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def concat(self, leftNet, rightNet, name):
        with tf.variable_scope(name):
            sameShape = False
            if (leftNet.shape.rank == rightNet.shape.rank) :
                sameShape = True
                for i in range(0,leftNet.shape.rank):
                    if leftNet.shape.dims[i].value != rightNet.shape.dims[i].value:
                        sameShape = False
                        break

            assert sameShape
            concat = tf.concat([leftNet,rightNet],3)
            return concat

    def sum(self, leftNet, rightNet, name):
        with tf.variable_scope(name):
            sameShape = False
            if (leftNet.shape.rank == rightNet.shape.rank) :
                sameShape = True
                for i in range(0,leftNet.shape.rank):
                    if leftNet.shape.dims[i].value != rightNet.shape.dims[i].value:
                        sameShape = False
                        break

            assert sameShape
            sum = tf.nn.relu6(leftNet + rightNet)
            return sum

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

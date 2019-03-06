# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import tensorflow as tf
import numpy as np

def _conv(x, kernel_size, out_channels, stride, var_list, pad="SAME", name="conv"):
    """
    Define API for conv operation. This includes kernel declaration and
    conv operation both.
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        n = kernel_size * kernel_size * out_channels
        # n = kernel_size * in_channels
        # stdv = 1.0 / math.sqrt(n)
        w = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels],
                           tf.float32, 
                           # initializer=tf.random_uniform_initializer(-stdv, stdv))
                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))

        # Append the variable to the trainable variables list
        var_list.append(w)

    # Do the convolution operation
    output = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=pad)
    return output

def _fc(x, out_dim, var_list, name="fc", is_cifar=False):
    """
    Define API for the fully connected layer. This includes both the variable
    declaration and matmul operation.
    """
    in_dim = x.get_shape().as_list()[1]
    stdv = 1.0 / math.sqrt(in_dim)
    with tf.variable_scope(name):
        # Define the weights and biases for this layer
        w = tf.get_variable('weights', [in_dim, out_dim], tf.float32,
                            initializer=tf.initializers.variance_scaling(scale=1.0, distribution='uniform'))
                #initializer=tf.random_uniform_initializer(-stdv, stdv))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))

        b = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.constant_initializer(0))

        # Append the variable to the trainable variables list
        var_list.append(w)
        var_list.append(b)

    # Do the FC operation
    output = tf.nn.xw_plus_b(x, w, b)
    return output

def _bn(x, var_list, train_phase, name='bn_'):
    """
    Batch normalization on convolutional maps.
    Args:

    Return:
    """
    n_out = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        var_list.append(beta)
        var_list.append(gamma)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

def _residual_block(x, trainable_vars, train_phase, apply_relu=True, leakiness=0.0, name="unit"):
    """
    ResNet block when the number of channels across the skip connections are the same
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        shortcut = x
        x = _conv(x, 3, in_channels, 1, trainable_vars, name='conv_1')
        x = _bn(x, trainable_vars, train_phase, name="bn_1")
        x = tf.nn.leaky_relu(x, alpha=leakiness)
        x = _conv(x, 3, in_channels, 1, trainable_vars, name='conv_2')
        x = _bn(x, trainable_vars, train_phase, name="bn_2")

        x = x + shortcut
        if apply_relu == True:
            x = tf.nn.leaky_relu(x, alpha=leakiness)

    return x

def _residual_block_first(x, out_channels, strides, trainable_vars, train_phase, apply_relu=True, leakiness=0.0, name="unit", is_ATT_DATASET=False):
    """
    A generic ResNet Block
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        # Figure out the shortcut connection first
        if in_channels == out_channels:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channels, strides, trainable_vars, name="shortcut")
            if not is_ATT_DATASET:
                shortcut = _bn(shortcut, trainable_vars, train_phase, name="bn_0")

        # Residual block
        x = _conv(x, 3, out_channels, strides, trainable_vars, name="conv_1")
        x = _bn(x, trainable_vars, train_phase, name="bn_1")
        x = tf.nn.leaky_relu(x, alpha=leakiness)
        x = _conv(x, 3, out_channels, 1, trainable_vars, name="conv_2")
        x = _bn(x, trainable_vars, train_phase, name="bn_2")

        x = x + shortcut
        if apply_relu:
            x = tf.nn.leaky_relu(x, alpha=leakiness)

    return x

def _residual_block_downsample(x, out_channels, strides, trainable_vars, train_phase, apply_relu=True, leakiness=0.0, name="unit", is_ATT_DATASET=False):
    """
    A generic ResNet Block
    """
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        # Figure out the shortcut connection first
        if in_channels == out_channels:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channels, strides, trainable_vars, name="shortcut")
            if not is_ATT_DATASET:
                shortcut = _bn(shortcut, trainable_vars, train_phase, name="bn_0")

        # Residual block
        x = _conv(x, 3, out_channels, strides, trainable_vars, name="conv_1")
        x = _bn(x, trainable_vars, train_phase, name="bn_1")
        x = tf.nn.leaky_relu(x, alpha=leakiness)
        x = _conv(x, 3, out_channels, 1, trainable_vars, name="conv_2")
        x = _bn(x, trainable_vars, train_phase, name="bn_2")

        x = x + shortcut
        if apply_relu:
            x = tf.nn.leaky_relu(x, alpha=leakiness)

    return x

def _residual(x, out_channels, stride, trainable_vars, train_phase, leakiness=0.0, name="unit",
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    in_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        if activate_before_residual:
          with tf.variable_scope('shared_activation'):
            x = _bn(x,trainable_vars,train_phase,'init_bn')
            x =  tf.nn.leaky_relu(x, alpha=leakiness)
            orig_x = x
        else:
          with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = _bn(x, trainable_vars, train_phase, 'init_bn')
            x = tf.nn.leaky_relu(x, alpha=leakiness)

        with tf.variable_scope('sub1'):
          x = _conv(x, 3, out_channels, stride, trainable_vars, name="conv1")

        with tf.variable_scope('sub2'):
          x = _bn(x, trainable_vars, train_phase, 'bn2')
          x =  tf.nn.leaky_relu(x, alpha=leakiness)
          x = _conv(x, 3, out_channels, 1, trainable_vars, name="conv2")

        with tf.variable_scope('sub_add'):
          if in_channels != out_channels:
            orig_x = tf.nn.avg_pool(orig_x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_channels-in_channels)//2, (out_channels-in_channels)//2]])
          x += orig_x
    return x
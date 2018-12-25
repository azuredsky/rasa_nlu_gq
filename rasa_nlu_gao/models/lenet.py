from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
import math

def cosine_losses(embedding, labels, out_num, s=30., m=0.4, w_init=None):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        #embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        #embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')
        inv_mask = tf.subtract(1., labels, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, labels), name='cosine_loss_output')
    return output

def arcface_loss(embedding, labels, out_num, s=64., m=0.5, w_init=None):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        #mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., labels, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, labels), name='arcface_loss_output')
    return output

def cnn_single_layer(x, is_training):
    pooled_outputs = []
    sequence_length = 25
    filter_sizes = [1,3,5,11]
    num_filters = len(filter_sizes)
    num_filters_total = sum(filter_sizes)
    for i, filter_size in enumerate(filter_sizes):
        # with tf.name_scope("convolution-pooling-%s" %filter_size):
        with tf.variable_scope("convolution-pooling-%s" % filter_size):
            # ====>a.create filter
            filter = tf.get_variable("filter-%s" % filter_size, [filter_size, x.shape[-1], 1, num_filters])

            conv = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
            conv = tf.contrib.layers.batch_norm(conv, is_training=is_training, scope='cnn_bn_')
            print(i,conv)

            b = tf.get_variable("b-%s" % filter_size, [num_filters])  # ADD 2017-06-09
            h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
            pooled_outputs.append(pooled)
    print(pooled_outputs)
    h_pool = tf.concat(pooled_outputs,3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
    print(h_pool)
    return h_pool

# Build a convolutional neural network
def conv_net(x, labels, n_classes, num_layers,layer_size, C2, dropout, scale, margin, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(x.shape[-1]) #forward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(x.shape[-1]) #backward direction cell

        encoder_f_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=0.5)
        encoder_b_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=0.5)

        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(encoder_f_cell, encoder_b_cell, x, dtype=tf.float32)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2) #[batch_size,sequence_length,hidden_size*2]
        encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
        encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        print(encoder_outputs, encoder_final_state_c, encoder_final_state_h)
        #encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

        reg = tf.contrib.layers.l2_regularizer(C2)

        name = 'dense'
        for i in range(num_layers):
            x = tf.layers.dense(inputs=encoder_final_state_h,
                                units=layer_size[i],
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i))
            x = tf.layers.dropout(x, rate=dropout, training=is_training)


        #'''
        out = tf.layers.dense(inputs=x,
                            units=n_classes,
                            kernel_regularizer=reg,
                            name='dense_layer_{}'.format(name))
        #'''

    #out = cosine_losses(x, labels, n_classes, scale, margin)

    return out
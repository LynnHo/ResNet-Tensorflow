from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf

from tensorflow.contrib import slim


DEFAULT_IMAGE_SIZE = 224


def residual_block(inputs,
                   num_out_channels,
                   num_layers,
                   num_filters,
                   is_down_sample,
                   is_pre_act=True,
                   activation_fn=tf.nn.relu,
                   normalizer_fn=None,
                   normalizer_params=None,
                   weights_initializer=slim.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=tf.constant_initializer(0.2),
                   biases_regularizer=None,
                   reuse=None,
                   trainable=True,
                   scope=None):
    assert num_layers in [2, 3], '`num_layers` should be 2 or 3!'

    conv = partial(slim.conv2d,
                   activation_fn=activation_fn,
                   normalizer_fn=normalizer_fn,
                   normalizer_params=normalizer_params,
                   weights_initializer=weights_initializer,
                   weights_regularizer=weights_regularizer,
                   biases_initializer=biases_initializer,
                   biases_regularizer=biases_regularizer,
                   trainable=trainable)
    stride = 2 if is_down_sample else 1

    def _residual_fn(inputs):
        if num_layers == 2:
            residual = conv(inputs, num_filters, 3, stride)
            residual = conv(residual, num_out_channels, 3, 1, activation_fn=None, normalizer_fn=None)
        elif num_layers == 3:
            residual = conv(inputs, num_filters, 1, 1)  # stride in the first or the second conv?
            residual = conv(residual, num_filters, 3, stride)
            residual = conv(residual, num_out_channels, 1, 1, activation_fn=None, normalizer_fn=None)
        return residual

    with tf.variable_scope(scope, 'residual_block_%d' % num_layers, [inputs], reuse=reuse):
        if is_pre_act:
            pre_act = activation_fn(normalizer_fn(inputs, **normalizer_params))
        else:
            pre_act = inputs

        in_channels = inputs.get_shape().as_list()[-1]
        if not is_down_sample and in_channels == num_out_channels:
            shortcut = inputs
        else:
            # shortcut = conv(inputs, num_out_channels, 1, stride, activation_fn=None, normalizer_fn=None)
            shortcut = conv(pre_act, num_out_channels, 1, stride, activation_fn=None, normalizer_fn=None)

        residual = _residual_fn(pre_act)

        return shortcut + residual

residual_block_2 = partial(residual_block, num_layers=2)
residual_block_3 = partial(residual_block, num_layers=3)


def resnet_basic(inputs,
                 dim_outputs=1000,
                 dim_conv1=64,
                 dims_conv2_to_5=[64, 128, 256, 512],
                 repeats_conv2_to_5=[2, 2, 2, 2],
                 block_fn=residual_block_2,  # resnet_18
                 weights_regularizer=None,
                 is_training=True,
                 reuse=None,
                 updates_collections=None,
                 scope=None):
    assert isinstance(dims_conv2_to_5, (list, tuple)) and len(dims_conv2_to_5) == 4, \
        '`dims_conv2_to_5` should be a list/tuple of length 4!'
    assert isinstance(repeats_conv2_to_5, (list, tuple)) and len(repeats_conv2_to_5) == 4, \
        '`repeats_conv2_to_5` should be a list/tuple of length 4!'
    assert block_fn in [residual_block_2, residual_block_3], 'Unavailable `block_fn`!'

    norm = partial(slim.batch_norm,
                   scale=True,
                   epsilon=1e-5,
                   is_training=is_training,
                   updates_collections=updates_collections)

    def _res_block(x, d, is_down_sample=False, is_pre_act=True):
        if block_fn == residual_block_2:
            num_out_channels = d
        elif block_fn == residual_block_3:
            num_out_channels = 4 * d
        return block_fn(x,
                        num_out_channels=num_out_channels,
                        num_filters=d,
                        is_down_sample=is_down_sample,
                        is_pre_act=is_pre_act,
                        normalizer_fn=norm,
                        weights_regularizer=weights_regularizer)

    with tf.variable_scope(scope, 'resnet_basic', [inputs], reuse=reuse):
        # conv1
        net = slim.conv2d(inputs, dim_conv1, 7, 2,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=norm,
                          weights_regularizer=weights_regularizer)

        # conv2_x ~ conv5_x
        for i, (dim, repeat) in enumerate(zip(dims_conv2_to_5, repeats_conv2_to_5)):
            if i == 0:
                net = slim.max_pool2d(net, 3, 2, padding='SAME')
                net = _res_block(net, dim, is_pre_act=False)
                for _ in range(repeat - 1):
                    net = _res_block(net, dim)
            else:
                net = _res_block(net, dim, is_down_sample=True)
                for _ in range(repeat - 1):
                    net = _res_block(net, dim)

        # fc
        net = tf.reduce_mean(net, axis=[2, 3])
        net = slim.fully_connected(net, dim_outputs,
                                   activation_fn=None,
                                   weights_regularizer=weights_regularizer)

        return net


def resnet(inputs,
           resnet_type='resnet_18',
           dim_outputs=1000,
           dim=64,
           weights_regularizer=None,
           is_training=True,
           reuse=None,
           updates_collections=None,
           scope=None):
    resnet_params = {'resnet_18': dict(repeats_conv2_to_5=[2, 2, 2, 2], block_fn=residual_block_2),
                     'resnet_34': dict(repeats_conv2_to_5=[3, 4, 6, 3], block_fn=residual_block_2),
                     'resnet_50': dict(repeats_conv2_to_5=[3, 4, 6, 3], block_fn=residual_block_3),
                     'resnet_101': dict(repeats_conv2_to_5=[3, 4, 23, 3], block_fn=residual_block_3),
                     'resnet_152': dict(repeats_conv2_to_5=[3, 8, 36, 3], block_fn=residual_block_3)}

    assert resnet_type in resnet_params, 'Unavailable `resnet_type`!'

    return resnet_basic(inputs=inputs,
                        dim_outputs=dim_outputs,
                        dim_conv1=dim,
                        dims_conv2_to_5=[dim, dim * 2, dim * 4, dim * 8],
                        weights_regularizer=weights_regularizer,
                        is_training=is_training,
                        reuse=reuse,
                        updates_collections=updates_collections,
                        scope=scope,
                        **resnet_params[resnet_type])

# alis
resnet_18 = partial(resnet, resnet_type='resnet_18')
resnet_34 = partial(resnet, resnet_type='resnet_34')
resnet_50 = partial(resnet, resnet_type='resnet_50')
resnet_101 = partial(resnet, resnet_type='resnet_101')
resnet_152 = partial(resnet, resnet_type='resnet_152')

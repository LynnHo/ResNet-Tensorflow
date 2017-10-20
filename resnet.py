from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim


# @TODO - Pre-activation residual_block
# @TODO - ResNeXt


default_image_size = 224

conv2d = slim.conv2d
fully_connected = slim.fully_connected
max_pool2d = slim.max_pool2d
avg_pool2d = slim.avg_pool2d
batch_norm = slim.batch_norm
dropout = slim.dropout


def residual_block_2(inputs,
                     out_channels,
                     num_filters,
                     resample=None,
                     activation_fn=tf.nn.relu,
                     normalizer_fn=None,
                     normalizer_params=None,
                     weights_initializer=slim.xavier_initializer(),
                     weights_regularizer=None,
                     biases_initializer=tf.constant_initializer(0.2),
                     biases_regularizer=None,
                     scope=None):
    """Residual block.

    resample: None, 'down' or 'up'.
    """
    assert resample in [None, 'down', 'up'], \
        "`resample` should be None, 'down' or 'up'!"

    params = dict(activation_fn=activation_fn,
                  normalizer_fn=normalizer_fn,
                  normalizer_params=normalizer_params,
                  weights_initializer=weights_initializer,
                  weights_regularizer=weights_regularizer,
                  biases_initializer=biases_initializer,
                  biases_regularizer=biases_regularizer)

    conv = functools.partial(slim.conv2d, **params)
    deconv = functools.partial(slim.conv2d_transpose, **params)

    with tf.variable_scope(scope, 'residual_block_2', [inputs]):
        in_channels = inputs.get_shape().as_list()[-1]

        if resample is None:
            if in_channels == out_channels:
                shortcut = inputs
            else:
                shortcut = conv(inputs, out_channels, 1, 1, activation_fn=None)
            residual = conv(inputs, num_filters, 3, 1)
            residual = conv(residual, out_channels, 3, 1, activation_fn=None)

        elif resample == 'down':
            shortcut = conv(inputs, out_channels, 1, 2, activation_fn=None)
            residual = conv(inputs, num_filters, 3, 2)
            residual = conv(residual, out_channels, 3, 1, activation_fn=None)

        elif resample == 'up':
            shortcut = deconv(inputs, out_channels, 1, 2, activation_fn=None)
            residual = deconv(inputs, num_filters, 3, 2)
            residual = conv(residual, out_channels, 3, 1, activation_fn=None)

        return activation_fn(shortcut + residual)


def residual_block_3(inputs,
                     out_channels,
                     num_filters,
                     resample=None,
                     activation_fn=tf.nn.relu,
                     normalizer_fn=None,
                     normalizer_params=None,
                     weights_initializer=slim.xavier_initializer(),
                     weights_regularizer=None,
                     biases_initializer=tf.constant_initializer(0.2),
                     biases_regularizer=None,
                     scope=None):
    """Residual block.

    resample: None, 'down' or 'up'.
    """
    assert resample in [None, 'down', 'up'], \
        "`resample` should be None, 'down' or 'up'!"

    params = dict(activation_fn=activation_fn,
                  normalizer_fn=normalizer_fn,
                  normalizer_params=normalizer_params,
                  weights_initializer=weights_initializer,
                  weights_regularizer=weights_regularizer,
                  biases_initializer=biases_initializer,
                  biases_regularizer=biases_regularizer)

    conv = functools.partial(slim.conv2d, **params)
    deconv = functools.partial(slim.conv2d_transpose, **params)

    with tf.variable_scope(scope, 'residual_block_3', [inputs]):
        in_channels = inputs.get_shape().as_list()[-1]

        if resample is None:
            if in_channels == out_channels:
                shortcut = inputs
            else:
                shortcut = conv(inputs, out_channels, 1, 1, activation_fn=None)
            residual = conv(inputs, num_filters, 1, 1)
            residual = conv(residual, num_filters, 3, 1)
            residual = conv(residual, out_channels, 1, 1, activation_fn=None)

        elif resample == 'down':
            shortcut = conv(inputs, out_channels, 1, 2, activation_fn=None)
            residual = conv(inputs, num_filters, 1, 2)
            residual = conv(residual, num_filters, 3, 1)
            residual = conv(residual, out_channels, 1, 1, activation_fn=None)

        elif resample == 'up':
            shortcut = deconv(inputs, out_channels, 1, 2, activation_fn=None)
            residual = deconv(inputs, num_filters, 1, 2)
            residual = conv(residual, num_filters, 3, 1)
            residual = conv(residual, out_channels, 1, 1, activation_fn=None)

        return activation_fn(shortcut + residual)


def resnet_basic(inputs,
                 num_outputs=1000,
                 dim_conv1=64,
                 dims_conv2_to_5=[64, 128, 256, 512],
                 repeats_conv2_to_5=[2, 2, 2, 2],
                 block_fn=residual_block_2,  # resnet_18
                 weights_regularizer=None,
                 is_training=True,
                 reuse=False,
                 updates_collections=None,
                 scope=None):
    assert (isinstance(dims_conv2_to_5, (list, tuple)) and
            len(dims_conv2_to_5) == 4),\
        '`dims_conv2_to_5` should be a list(tuple) of length 4.'
    assert (isinstance(repeats_conv2_to_5, (list, tuple)) and
            len(repeats_conv2_to_5) == 4),\
        '`repeats_conv2_to_5` should be a list(tuple) of length 4.'
    assert block_fn in [residual_block_2, residual_block_3], 'Block type error!'

    # deal with difference
    norm = functools.partial(batch_norm,
                             scale=True,
                             epsilon=1e-5,
                             is_training=is_training,
                             updates_collections=updates_collections)
    res_block_ = functools.partial(block_fn,
                                   normalizer_fn=norm,
                                   weights_regularizer=weights_regularizer,
                                   biases_initializer=None)

    def res_block(x, d, resample=None):
        if block_fn == residual_block_2:
            return res_block_(x, d, d, resample=resample)
        elif block_fn == residual_block_3:
            return res_block_(x, d * 4, d, resample=resample)

    # unified part
    with tf.variable_scope(scope, 'resnet_basic', [inputs], reuse=reuse):
        # conv1
        net = conv2d(inputs, dim_conv1, 7, 2,
                     biases_initializer=None,
                     activation_fn=tf.nn.relu,
                     normalizer_fn=norm)

        # conv2_x ~ conv5_x
        for dim, repeat, i in zip(dims_conv2_to_5,
                                  repeats_conv2_to_5,
                                  range(len(dims_conv2_to_5))):
            if i == 0:
                net = max_pool2d(net, 3, 2, padding='SAME')
                for r in range(repeat):
                    net = res_block(net, dim)
            else:
                net = res_block(net, dim, resample='down')
                for r in range(repeat - 1):
                    net = res_block(net, dim)

        # fc
        net = avg_pool2d(net, 7, 1)
        net = slim.flatten(net)
        net = fully_connected(net, num_outputs,
                              activation_fn=None,
                              weights_regularizer=weights_regularizer)

        return net


def resnet(inputs,
           num_outputs=1000,
           dim=64,
           weights_regularizer=None,
           is_training=True,
           reuse=False,
           updates_collections=None,
           scope=None,
           resnet_type='resnet_18'):
    resnet_params = {'resnet_18': dict(repeats_conv2_to_5=[2, 2, 2, 2],
                                       block_fn=residual_block_2),
                     'resnet_34': dict(repeats_conv2_to_5=[3, 4, 6, 3],
                                       block_fn=residual_block_2),
                     'resnet_50': dict(repeats_conv2_to_5=[3, 4, 6, 3],
                                       block_fn=residual_block_3),
                     'resnet_101': dict(repeats_conv2_to_5=[3, 4, 23, 3],
                                        block_fn=residual_block_3),
                     'resnet_152': dict(repeats_conv2_to_5=[3, 8, 36, 3],
                                        block_fn=residual_block_3)}

    assert resnet_type in resnet_params, 'Resnet type error!'

    scope = resnet_type if scope is None else scope

    return resnet_basic(inputs=inputs,
                        num_outputs=num_outputs,
                        dim_conv1=dim,
                        dims_conv2_to_5=[dim, dim * 2, dim * 4, dim * 8],
                        weights_regularizer=weights_regularizer,
                        is_training=is_training,
                        reuse=reuse,
                        updates_collections=updates_collections,
                        scope=scope,
                        **resnet_params[resnet_type])

# alis
resnet_18 = functools.partial(resnet, resnet_type='resnet_18')
resnet_34 = functools.partial(resnet, resnet_type='resnet_34')
resnet_50 = functools.partial(resnet, resnet_type='resnet_50')
resnet_101 = functools.partial(resnet, resnet_type='resnet_101')
resnet_152 = functools.partial(resnet, resnet_type='resnet_152')

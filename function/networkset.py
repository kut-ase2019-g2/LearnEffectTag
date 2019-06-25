import sys
import tensorflow.compat.v1 as tf
import net_function


def run(netset, input, height, width, dropout=False, padding=0):

    # =====encode_2-valid================================================================================
    if netset == 'encode_2-morech-valid44':
        layers = []
        sample_data = tf.reshape(input, shape=[-1, height, width, 3])
        layers.append(sample_data)
        with tf.variable_scope('conv_1'):
            conv_1 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 3, 60], 1)
            layers.append(conv_1)
        with tf.variable_scope('conv_2'):
            conv_2 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 60, 120], 2)
            layers.append(conv_2)
        with tf.variable_scope('conv_3'):
            conv_3 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 120, 120], 1)
            layers.append(conv_3)
        with tf.variable_scope('conv_4'):
            conv_4 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 120, 240], 2)
            layers.append(conv_4)
        with tf.variable_scope('conv_5'):
            conv_5 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_5)
        with tf.variable_scope('conv_6'):
            conv_6 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_6)
        with tf.variable_scope('conv_7'):
            conv_7 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_7)
        with tf.variable_scope('conv_8'):
            conv_8 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_8)
        with tf.variable_scope('conv_9'):
            conv_9 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_9)
        with tf.variable_scope('conv_10'):
            conv_10 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_10)
        with tf.variable_scope('conv_11'):
            conv_11 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_11)
        with tf.variable_scope('conv_12'):
            conv_12 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 240, 240], 1)
            layers.append(conv_12)

        with tf.variable_scope('conv_13'):
            resize_13 = tf.image.resize_images(layers[-1], [tf.shape(layers[-1])[1] * 2, tf.shape(layers[-1])[2] * 2],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv_13 = net_function.relu_conv_valid_layer(resize_13, [3, 3, 240, 120], 1)
            layers.append(conv_13)
        with tf.variable_scope('conv_14'):
            conv_14 = net_function.relu_conv_valid_layer(layers[-1], [3, 3, 120, 120], 1)
            layers.append(conv_14)
        with tf.variable_scope('conv_15'):
            resize_15 = tf.image.resize_images(layers[-1], [tf.shape(layers[-1])[1] * 2, tf.shape(layers[-1])[2] * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            conv_15 = net_function.relu_conv_valid_layer(resize_15, [3, 3, 120, 60], 1)
            layers.append(conv_15)
        with tf.variable_scope('conv_16'):
            conv_16_nonsig, conv_16_sig = net_function.sigmoid_conv_valid_layer(layers[-1], [3, 3, 60, 1], 1)
            layers.append(conv_16_nonsig)
            layers.append(conv_16_sig)
        output_nonsig = tf.reshape(layers[-2], [-1, (height-88) * (width-88) * 1])
        output_sig = tf.reshape(layers[-1], [-1, (height-88) * (width-88) * 1])
        return output_nonsig, output_sig

    print('=====================')
    print('Not Match NetSet!')
    print('Please Check NetSet!')
    print('=====================')
    sys.exit()

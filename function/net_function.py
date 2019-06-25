import tensorflow.compat.v1 as tf


# 重み生成
def Xavier_filter_weight_variable(shape, name=None):
    incoming = shape[0] * shape[1] * shape[2]  # フィルタ１つのサイズ
    weight_init = tf.random_normal_initializer(stddev=(1.0 / incoming) ** 0.5)  # 正規分布でテンソルを生成する stddev:生成する乱数の標準偏差
    W = tf.get_variable(name=name, shape=shape, initializer=weight_init, trainable=True)  # 層の重みを生成 initializer:初期化子
    return W


# ReLU用重み生成
def He_filter_weight_variable(shape, name=None):
    incoming = shape[0] * shape[1] * shape[2]  # フィルタ１つのサイズ
    weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming) ** 0.5)  # 正規分布でテンソルを生成する stddev:生成する乱数の標準偏差
    W = tf.get_variable(name=name, shape=shape, initializer=weight_init, trainable=True)  # 層の重みを生成 initializer:初期化子
    return W


# バイアス生成
def bias_variable(shape, name=None):
    incoming = shape[3]
    bias_init = tf.constant_initializer(value=0)  # 全て一定値のテンソルを生成する
    b = tf.get_variable(name=name, shape=incoming, initializer=bias_init, trainable=True)
    return b


# maxpool
def maxpooling(input, ksize=None):
    if ksize == None:
        ksize = [1, 2, 2, 1]
    out = tf.nn.max_pool(input, ksize=ksize, strides=[1, 2, 2, 1], padding='SAME')
    return out


# 畳み込み層+ReLU関数
def conv_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    return conv_normed

# 畳み込み層+ReLU関数
def relu_conv_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out = tf.nn.relu(conv_normed)
    return out


# 畳み込み層+ReLU関数
def relu_atrous_conv_layer(input, filter_shape, rate, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.atrous_conv2d(value=input, filters=filter, rate=rate, padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out = tf.nn.relu(conv_normed)
    return out


# 畳み込み層+ReLU関数
def relu_conv_dropout_layer(input, filter_shape, stride, non_drop=1, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    relu = tf.nn.relu(conv_normed)
    out = tf.nn.dropout(relu, non_drop)
    return out


# 畳み込み層+ReLU関数
def relu_conv_valid_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="VALID")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out = tf.nn.relu(conv_normed)
    return out


# 畳み込み層+ReLU関数
def relu_atrous_conv_valid_layer(input, filter_shape, rate, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.atrous_conv2d(value=input, filters=filter, rate=rate, padding="VALID")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out = tf.nn.relu(conv_normed)
    return out


# 畳み込み層+Sigmoid関数
def sigmoid_conv_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out_nonsig = conv_normed
    out_sig = tf.nn.sigmoid(conv_normed)
    return out_nonsig, out_sig


# 畳み込み層+Sigmoid関数
def sigmoid_conv_valid_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    filter = He_filter_weight_variable(filter_shape, weight_name)
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="VALID")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out_nonsig = conv_normed
    out_sig = tf.nn.sigmoid(conv_normed)
    return out_nonsig, out_sig


# 畳み込み層+ReLU関数
def relu_conv_transpose_layer(input, filter_shape, up_ration, stride, weight_name='W', bias_name='decode_b'):
    batchsize = tf.shape(input)[0]
    out_height = tf.shape(input)[1] * up_ration
    out_width = tf.shape(input)[2] * up_ration
    out_shape = [batchsize, out_height, out_width, filter_shape[3]]

    transpose_shape = [filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]]  # inとoutが逆
    filter = He_filter_weight_variable(transpose_shape, weight_name)
    transpose_filter = tf.transpose(filter, perm=[0, 1, 3, 2])
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d_transpose(input, filter=transpose_filter, output_shape=out_shape,
                                  strides=[1, stride, stride, 1], padding="VALID")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits)
    out = tf.nn.relu(conv_normed)
    return out

# 畳み込み層+ReLU関数
def relu_conv_transpose_valid_layer(input, filter_shape, up_ration, stride, weight_name='W', bias_name='decode_b'):
    batchsize = tf.shape(input)[0]
    out_height = (tf.shape(input)[1] * up_ration) + 2
    out_width = (tf.shape(input)[2] * up_ration) + 2
    out_shape = [batchsize, out_height, out_width, filter_shape[3]]

    transpose_shape = [filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]]  # inとoutが逆
    filter = He_filter_weight_variable(transpose_shape, weight_name)
    # transpose_filter = tf.transpose(filter, perm=[0, 1, 3, 2])
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d_transpose(input, filter=filter, output_shape=out_shape,
                                  strides=[1, stride, stride, 1], padding="VALID")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits, name='decode_conv')
    out = tf.nn.relu(conv_normed)
    return out


# deconvolution+ReLU関数
def relu_decode_conv_layer(input, filter_shape, stride, weight_name='W', bias_name='decode_b'):
    transpose_shape = [filter_shape[0], filter_shape[1], filter_shape[3], filter_shape[2]]  # inとoutが逆
    transpose_filter = He_filter_weight_variable(transpose_shape, weight_name=weight_name)
    filter = tf.transpose(transpose_filter, perm=[0, 1, 3, 2])
    bias = bias_variable(filter_shape, bias_name)
    conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="SAME")
    logits = tf.nn.bias_add(conv, bias)
    conv_normed = tf.layers.batch_normalization(logits, name='decode_conv')
    out = tf.nn.relu(conv_normed)
    return out


def residual_block(input, filter_shape_1, stride_1, filter_shape_2, stride_2, projection=False):
    input_depth = filter_shape_1[2]
    output_depth = filter_shape_2[3]
    conv1 = relu_conv_layer(input, filter_shape_1, stride_1, weight_name='W1', bias_name='b1')
    conv2 = relu_conv_layer(conv1, filter_shape_2, stride_2, weight_name='W2', bias_name='b2')

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = relu_conv_layer(input, [1, 1, input_depth, output_depth], 2, weight_name='W3', bias_name='b3')
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])

    else:
        input_layer = input

    res = conv2 + input_layer
    return res


def residual_block_valid(input, filter_shape_1, stride_1, filter_shape_2, stride_2, height, width, projection=False):
    input_depth = filter_shape_1[2]
    output_depth = filter_shape_2[3]
    h_filter_leng_1 = int( (filter_shape_1[0] - 1) / 2 )
    w_filter_leng_1 = int( (filter_shape_1[1] - 1) / 2 )
    h_filter_leng_2 = int( (filter_shape_2[0] - 1) / 2 )
    w_filter_leng_2 = int( (filter_shape_2[1] - 1) / 2 )
    # h_filter_leng_1 = 0
    # w_filter_leng_1 = 0
    # h_filter_leng_2 = 0
    # w_filter_leng_2 = 0
    h_filter_leng = h_filter_leng_1 + h_filter_leng_2
    w_filter_leng = w_filter_leng_1 + w_filter_leng_2
    conv1 = relu_conv_valid_layer(input, filter_shape_1, stride_1, weight_name='W1', bias_name='b1')
    conv2 = relu_conv_valid_layer(conv1, filter_shape_2, stride_2, weight_name='W2', bias_name='b2')

    # if input_depth != output_depth:
    #     if projection:
    #         # Option B: Projection shortcut
    #         input_layer = relu_conv_valid_layer(input, [1, 1, input_depth, output_depth], 2, weight_name='W3', bias_name='b3')
    #     else:
    #         # Option A: Zero-padding
    #         input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])

    # else:
    # _, height, width, _ = tf.shape(input)
    # input_layer = tf.image.crop_to_bounding_box(input, h_filter_leng, w_filter_leng, height-h_filter_leng, width-w_filter_leng)
    input_layer = tf.image.crop_to_bounding_box(input, h_filter_leng, w_filter_leng, height-(h_filter_leng*2), width-(w_filter_leng*2))

    # input_layer = input[-1, h_filter_leng: -h_filter_leng, w_filter_leng: -w_filter_leng, output_depth]
    # input_layer = input[-1, 1: -1, 1: -1, output_depth]


    res = conv2 + input_layer
    return res


    # # 畳み込み層+ReLU関数
    # def relu_conv_valid_layer(input, filter_shape, stride, weight_name='W', bias_name='b'):
    #     filter = He_filter_weight_variable(filter_shape, weight_name)
    #     bias = bias_variable(filter_shape, bias_name)
    #     conv = tf.nn.conv2d(input, filter=filter, strides=[1, stride, stride, 1], padding="VALID")
    #     logits = tf.nn.bias_add(conv, bias)
    #     conv_normed = tf.layers.batch_normalization(logits)
    #     out = tf.nn.relu(conv_normed)
    #     return out

def residual_block_noReLU_beforeBN(input, filter_shape_1, stride_1, filter_shape_2, stride_2, projection=False):
    input_depth = filter_shape_1[2]
    output_depth = filter_shape_2[3]

    filter_1 = He_filter_weight_variable(filter_shape_1, 'W1')
    bias_1 = bias_variable(filter_shape_1, 'b1')
    conv_1 = tf.nn.conv2d(input, filter=filter_1, strides=[1, stride_1, stride_1, 1], padding="SAME")
    logits_1 = tf.nn.bias_add(conv_1, bias_1)
    conv_normed_1 = tf.layers.batch_normalization(logits_1)
    out_1 = tf.nn.relu(conv_normed_1)

    filter_2 = He_filter_weight_variable(filter_shape_2, 'W2')
    bias_2 = bias_variable(filter_shape_2, 'b2')
    conv_2 = tf.nn.conv2d(out_1, filter=filter_2, strides=[1, stride_2, stride_2, 1], padding="SAME")
    logits_2 = tf.nn.bias_add(conv_2, bias_2)
    conv_normed_2 = tf.layers.batch_normalization(logits_2)
    out_2 = conv_normed_2

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = relu_conv_layer(input, [1, 1, input_depth, output_depth], 2, weight_name='W3', bias_name='b3')
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])

    else:
        input_layer = input

    res = out_2 + input_layer
    return res



# def residual_block(input, output_depth, down_sample, projection=False):
#     input_depth = input.get_shape().as_list()[3]
#     if down_sample:
#         conv1 = relu_conv_layer(input, [3, 3, input_depth, output_depth], 2)
#     else:
#         conv1 = relu_conv_layer(input, [3, 3, input_depth, output_depth], 1)
#     conv2 = relu_conv_layer(conv1, [3, 3, output_depth, output_depth], 1)
#
#     if input_depth != output_depth:
#         if projection:
#             # Option B: Projection shortcut
#             input_layer = relu_conv_layer(input, [1, 1, input_depth, output_depth], 2)
#         else:
#             # Option A: Zero-padding
#             input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
#
#     else:
#         input_layer = input
#
#     res = conv2 + input_layer
#     return res

import tensorflow as tf
import tensorflow.contrib.slim as slim
from util import log


def norm(x, norm_type, is_train, G=32, eps=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + eps)
            # per channel gamma and beta
            gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])

        elif norm_type == 'batch_mine':
            N, H, W, C = x.get_shape().as_list()

            batch_mean1, batch_var1 = tf.nn.moments(x, [0])

            # Apply the initial batch normalizing transform
            z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + eps)

            # Create two new parameters, scale and beta (shift)
            gamma = tf.Variable(tf.ones([C]), dtype=tf.float32, name='gamma')
            beta = tf.Variable(tf.zeros([C]), dtype=tf.float32, name='beta')

            # Scale and shift to obtain the final output of the batch normalization
            output = gamma * z1_hat + beta

        elif norm_type == 'batch_skew':
            N, H, W, C = x.get_shape().as_list()

            batch_mean1, batch_var1 = tf.nn.moments(x, [0])

            # Apply the initial batch normalizing transform
            z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + eps)

            # Create two new parameters, scale and beta (shift)
            gamma1 = tf.Variable(tf.ones(C), dtype=tf.float32, name='gamma1')
            gamma2 = tf.Variable(tf.ones([C]), dtype=tf.float32, name='gamma2')
            beta = tf.Variable(tf.zeros([C]), dtype=tf.float32, name='beta')

            print([*z1_hat.shape[:-1] , 1])
            print(gamma1.shape)

            ## Get tile to work later
            # tile1 = tf.tile(gamma1, [64,16,16,1] )
            # tile2 = tf.tile(gamma2, *z1_hat.shape[:-1] , 1 )

            ones = tf.ones_like(z1_hat)
            tile1 = gamma1 * ones
            tile2 = gamma2 * ones

            combined_gamma = tf.where(tf.less_equal(z1_hat,0), tile1, tile2) # # of matches, # of axes

            gamma_diff = gamma2 - gamma1
            # Scale and shift to obtain the final output of the batch normalization
            # this value is fed into the activation function (here a sigmoid)

            output = combined_gamma * z1_hat + beta

        else:
            raise NotImplementedError
    return output


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, info=False,
           activation_fn=lrelu, norm_type='batch',
           k=4, s=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s, s, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        if activation_fn is not None:
            activation = activation_fn(conv + biases)
        else:
            activation = conv + biases
        output = norm(activation, norm_type, is_train)
    if info: log.info('{} {}'.format(name, output))
    return output


def fc(input, output_shape, is_train, info=False,
       norm_type='batch', activation_fn=lrelu, name="fc"):
    activation = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    output = norm(activation, norm_type, is_train)
    if info: log.info('{} {}'.format(name, output))
    return output


def residual_block(input, output_shape, is_train, info=False, k=3, s=1,
                   name="residual", activation_fn=lrelu, norm_type='batch'):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            _ = conv2d(input, output_shape, is_train, k=k, s=s,
                       activation_fn=None, norm_type=norm_type)
            _ = norm(_, norm_type, is_train)
            _ = activation_fn(_)
        with tf.variable_scope('res2'):
            _ = conv2d(_, output_shape, is_train, k=k, s=s,
                       activation_fn=None, norm_type=norm_type)
            _ = norm(_, norm_type, is_train)
        _ = activation_fn(_ + input)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _

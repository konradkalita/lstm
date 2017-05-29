import tensorflow as tf


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


class EmbeddingWrapper:
    def __init__(self, cell, size, alphabet_size):
        self.cell = cell
        self.W_in = create_weights([alphabet_size, size])
        self.b_in = create_weights([size])
        self.b_out = create_weights([alphabet_size])

    def embedd_input(self, x_t):
        return tf.matmul(x_t, self.W_in) + self.b_in

    def embedd_output(self, out):
        W_out = tf.transpose(self.W_in)
        out = tf.matmul(out, W_out) + self.b_out
        return out

    def __call__(self, x_t, h_t, c_t):
        x_t = tf.matmul(x_t, self.W_in) + self.b_in
        h_t, c_t = self.cell(x_t, h_t, c_t)
        W_out = tf.transpose(self.W_in)
        h_t = tf.matmul(h_t, W_out) + self.b_out
        return h_t, c_t


class MultiLayer:
    def __init__(self, cells):
        pass

    def __cell__(self, x_t, state):
        results = []
        for (cell, (prev_h_t, prev_c_t)) in zip(self.cells, state):
            h_t, c_t = cell(x_t, prev_h_t, prev_c_t)
            results.append([h_t, c_t])
            x_t = h_t
        return x_t,


class DroputWrapper:
    def __init__(self, cell, dropout, mask_shape, scope):
        self.cell = cell
        self.dropout = dropout
        self.scope = scope
        self.layer_mask = self._create_mask(mask_shape)
        self.state_mask = self._create_mask(mask_shape)
        self.h_t = cell.h_t
        self.c_t = cell.c_t

    def _apply_dropout(self, v, mask):
        if self.dropout == 0.0:
            return v
        else:
            return tf.cond(self.scope.is_training, lambda: (v * mask) / (1 - self.dropout), lambda: v)

    def __call__(self, x_t, h_t, c_t):
        x_t = self._apply_dropout(x_t, self.layer_mask)
        h_t = self._apply_dropout(h_t, self.state_mask)
        h_t, c_t = self.cell(x_t, h_t, c_t)
        return h_t, c_t

    def _create_mask(self, shape):
        return tf.cast(tf.greater(tf.random_uniform(shape), self.dropout), tf.float32)

    def reset_mask(self, mask_shape):
        self.layer_mask = self._create_mask(mask_shape)
        self.state_mask = self._create_mask(mask_shape)


class LSTMCell:
    def __init__(self, size, input_size):
        self.h_t = tf.placeholder(tf.float32, shape=[None, size])
        self.c_t = tf.placeholder(tf.float32, shape=[None, size])
        self.W = [create_weights([size, size]) for i in range(4)]
        self.B = [create_weights([size]) for i in range(4)]

    def __call__(self, x_t, h_t, c_t):
        concat = h_t + x_t

        f_t = tf.sigmoid(tf.matmul(concat, self.W[0]) + (self.B[0] + 1))
        i_t = tf.sigmoid(tf.matmul(concat, self.W[1]) + self.B[1])
        c_prim_t = tf.tanh(tf.matmul(concat, self.W[2]) + self.B[2])

        c_t = f_t * c_t + i_t * c_prim_t

        o_t = tf.sigmoid(tf.matmul(concat, self.W[3]) + self.B[3])
        h_t = o_t * tf.tanh(c_t)

        return h_t, c_t


class NASCell:
    # based on NASCell in https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/rnn/python/ops/rnn_cell.py
    def __init__(self, size):
        self.h_t = tf.placeholder(tf.float32, shape=[None, size])
        self.c_t = tf.placeholder(tf.float32, shape=[None, size])
        self.W = [create_weights([size, size]) for i in range(16)]

    def __call__(self, x_t, h_t, c_t):
        # First layer

        l1_0 = tf.nn.relu(tf.matmul(x_t, self.W[0]) * tf.matmul(h_t, self.W[1]))
        l1_1 = tf.sigmoid(tf.matmul(x_t, self.W[2]) + tf.matmul(h_t, self.W[3]))
        l1_2 = tf.tanh(tf.matmul(x_t, self.W[4]) + tf.matmul(h_t, self.W[5]))
        l1_3 = tf.sigmoid(tf.matmul(x_t, self.W[6]) + tf.matmul(h_t, self.W[7]))
        l1_4 = tf.sigmoid(tf.matmul(x_t, self.W[8]) + tf.matmul(h_t, self.W[9]))
        l1_5 = tf.tanh(tf.matmul(x_t, self.W[10]) + tf.matmul(h_t, self.W[11]))
        l1_6 = tf.sigmoid(tf.matmul(x_t, self.W[12]) + tf.matmul(h_t, self.W[13]))
        l1_7 = tf.nn.relu(tf.matmul(x_t, self.W[14]) + tf.matmul(h_t, self.W[15]))

        # Second layer
        l2_0 = tf.sigmoid(l1_1 + l1_2)
        l2_1 = tf.tanh(l1_0 + l1_3)
        l2_2 = tf.tanh(l1_4 * l1_5)
        l2_3 = tf.tanh(l1_6 * l1_7)

        # Third layer
        l3_0 = tf.tanh(l2_0 + l2_2)
        l3_1 = tf.tanh(l2_3 + c_t)

        c_t = l2_1 * l3_1

        h_t = tf.tanh(l3_0 * c_t)

        return h_t, c_t

import tensorflow as tf


def initialize_matrix(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


class NASCell:
    def __init__(self, size):
        self.h_t = tf.placeholder(tf.float32, shape=[None, size])
        self.c_t = tf.placeholder(tf.float32, shape=[None, size])
        self.W = [initialize_matrix([size, size]) for i in range(32)]

    def step(self, x_t, h_t, c_t):
        a_0 = tf.nn.relu(tf.matmul(x_t, self.W[0]) * tf.matmul(h_t, self.W[1]))
        a_1 = tf.sigmoid(tf.matmul(x_t, self.W[2]) + tf.matmul(h_t, self.W[3]))
        a_2 = tf.tanh(tf.matmul(x_t, self.W[4]) + tf.matmul(h_t, self.W[5]))
        a_3 = tf.sigmoid(tf.matmul(x_t, self.W[6]) + tf.matmul(h_t, self.W[7]))
        a_4 = tf.sigmoid(tf.matmul(x_t, self.W[8]) + tf.matmul(h_t, self.W[9]))
        a_5 = tf.tanh(tf.matmul(x_t, self.W[10]) + tf.matmul(h_t, self.W[11]))
        a_6 = tf.sigmoid(tf.matmul(x_t, self.W[12]) + tf.matmul(h_t, self.W[13]))
        a_7 = tf.nn.relu(tf.matmul(x_t, self.W[14]) + tf.matmul(h_t, self.W[15]))

        b_0 = tf.sigmoid(a_1 + a_2)
        b_1 = tf.tanh(a_0 + a_3)
        b_2 = tf.tanh(a_4 * a_5)
        b_3 = tf.tanh(a_6 * a_7)

        c_4 = tf.tanh(b_0 + b_2)
        c_5 = tf.tanh(b_3 + c_t)

        c_t = b_1 * c_5

        h_t = tf.tanh(c_4 * c_t)

        return [h_t, c_t]


class LSTMCell:
    def __init__(self, size, input_size):
        print(size, input_size)
        self.h_t = tf.placeholder(tf.float32, shape=[None, size])
        self.c_t = tf.placeholder(tf.float32, shape=[None, size])

        self.W_in = initialize_matrix([input_size, size])
        self.b_in = initialize_matrix([size])

        self.W_f = initialize_matrix([size, size])
        self.b_f = initialize_matrix([size])

        self.W_i = initialize_matrix([size, size])
        self.b_i = initialize_matrix([size])

        self.W_c = initialize_matrix([size, size])
        self.b_c = initialize_matrix([size])

        self.W_o = initialize_matrix([size, size])
        self.b_o = initialize_matrix([size])

    def step(self, x_t, h_t, c_t):
        concat = h_t + x_t

        f_t = tf.sigmoid(tf.matmul(concat, self.W_f) + (self.b_f + 1))
        i_t = tf.sigmoid(tf.matmul(concat, self.W_i) + self.b_i)
        c_prim_t = tf.tanh(tf.matmul(concat, self.W_c) + self.b_c)

        c_t = f_t * c_t + i_t * c_prim_t

        o_t = tf.sigmoid(tf.matmul(concat, self.W_o) + self.b_o)
        h_t = o_t * tf.tanh(c_t)

        return [h_t, c_t]

import autograd.numpy as np


import tensorflow as tf

from cells import LSTMCell, NASCell


def initialize_matrix(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


class LSTM:

    def __init__(self, size, alphabet_size, batch_size, num_layers, dropout, cell_kind, num_steps, learning_rate):
        self.learning_rate = 1.0
        self.alphabet_size = alphabet_size
        self.batch_size = batch_size
        self.dropout_r = dropout
        self.num_layers = num_layers
        self.cell_kind = cell_kind
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.size = size
        self.build_model()
        self.embedding_nodes()
        self.construct_all()

    def create_dropout_mask(self, shape):
        result = tf.to_int64(tf.greater(tf.random_uniform(shape), self.dropout_r))
        return tf.cast(result, tf.float32)

    def build_model(self):
        if self.cell_kind:
            self.cells = [LSTMCell(self.size, self.alphabet_size) for i in range(self.num_layers)]
        else:
            self.cells = [NASCell(self.size) for i in range(self.num_layers)]
        self.apply_dropout = tf.placeholder(tf.bool)

    def dropout(self, v, mask):
        if self.dropout_r == 0.0:
            return v
        else:
            return tf.cond(self.apply_dropout, lambda: (v * mask) / (1 - self.dropout_r), lambda: v)

    def embedding_nodes(self):
        self.W_in = initialize_matrix([self.alphabet_size, self.size])
        self.b_in = initialize_matrix([self.size])
        self.b_out = initialize_matrix([self.alphabet_size])

    def embedd_input(self, x_t):
        return tf.matmul(x_t, self.W_in) + self.b_in

    def embedd_output(self, out):
        W_out = tf.transpose(self.W_in)
        out = tf.matmul(out, W_out) + self.b_out
        return out

    def step(self, x_t, prev_results, cells, masks):
        x_t = self.embedd_input(x_t)

        results = []
        for i in range(len(cells)):
            # vertical dropout
            x_t = self.dropout(x_t, masks[i])
            res = cells[i].step(x_t, prev_results[i][0], prev_results[i][1])
            results.append(res)
            x_t = res[0]
        y = self.embedd_output(x_t)
        y = tf.nn.softmax(y)
        return (y, results)

    def cost_function(self, cells, inputs, labels):
        horizontal_masks = [self.create_dropout_mask((self.batch_size, self.size)) for cell in cells]
        vertical_masks = [self.create_dropout_mask((self.batch_size, self.size)) for cell in cells]

        prev_results = [[self.dropout(cell.h_t, horizontal_masks[0]), self.dropout(cell.c_t, horizontal_masks[0])] for cell in cells]

        loss = tf.constant([0.0])

        predicts = []
        for i in range(self.batch_size):
            y, prev_results = self.step(inputs[i], prev_results, cells, vertical_masks)
            if i == 0:  # needed part of graph for generating text
                self.first_memory = prev_results

            for r in range(len(prev_results)):
                prev_results[r][0] = self.dropout(prev_results[r][0], horizontal_masks[r])

            predicts.append(y)
            loss += tf.reduce_sum(tf.log(y) * labels[i])

        loss = loss / (self.batch_size * self.num_steps)
        self.predicts = predicts
        self.memories = prev_results
        self.loss = -tf.reduce_sum(loss)
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def construct_all(self):
        self.batch_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.alphabet_size])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.alphabet_size])
        self.cost_function(self.cells, self.batch_placeholder, self.labels_placeholder)

    def reset_memories_dict(self, batch_size):
        result = {}
        for cell in self.cells:
            result[cell.h_t] = np.zeros((batch_size, self.size))
            result[cell.c_t] = np.zeros((batch_size, self.size))
        return result

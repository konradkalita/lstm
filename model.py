import autograd.numpy as np


import tensorflow as tf

from cells import LSTMCell, NASCell, DroputWrapper


def initialize_matrix(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


class Scope:
    pass


class LSTMModel:

    def __init__(self, config):
        self.alphabet_size = config.alphabet_size
        self.batch_size = config.batch_size
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        self.cell_kind = config.cell_kind
        self.num_steps = config.num_steps
        self.learning_rate = config.learning_rate
        self.scope = Scope()
        self.scope.is_training = tf.placeholder(tf.bool)
        self.size = config.size
        self.cells = [self.create_cell() for i in range(self.num_layers)]
        self.W_in = initialize_matrix([self.alphabet_size, self.size])
        self.b_in = initialize_matrix([self.size])
        self.b_out = initialize_matrix([self.alphabet_size])
        self.batch = tf.placeholder(tf.float32, shape=[None, None, self.alphabet_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, None, self.alphabet_size])
        self.cost_function(self.cells, self.batch, self.labels)

    def create_cell(self):
        if self.cell_kind:
            cell = LSTMCell(self.size, self.alphabet_size)
        else:
            cell = NASCell(self.size)
        cell = DroputWrapper(cell, self.dropout, (self.batch_size, self.size), self.scope)
        return cell

    def create_mask(self, shape):
        return tf.cast(tf.greater(tf.random_uniform(shape), self.dropout), tf.float32)

    def embedd_input(self, x_t):
        return tf.matmul(x_t, self.W_in) + self.b_in

    def embedd_output(self, out):
        W_out = tf.transpose(self.W_in)
        out = tf.matmul(out, W_out) + self.b_out
        return out

    def step(self, x_t, prev_states, cells):
        results = []
        x_t = self.embedd_input(x_t)
        for (cell, (prev_h_t, prev_c_t)) in zip(cells, prev_states):
            h_t, c_t = cell(x_t, prev_h_t, prev_c_t)
            results.append([h_t, c_t])
            x_t = h_t
        y = self.embedd_output(x_t)
        y = tf.nn.softmax(y)
        return (y, results)

    def cost_function(self, cells, inputs, labels):
        for cell in cells:
            cell.reset_mask((self.batch_size, self.size))
        prev_results = [[cell.h_t, cell.c_t] for cell in cells]
        loss = tf.constant([0.0])
        predicts = []
        for i in range(self.batch_size):
            y, prev_results = self.step(inputs[i], prev_results, cells)
            if i == 0:
                self.first_memory = prev_results

            predicts.append(y)
            loss += tf.reduce_sum(tf.log(y) * labels[i])

        loss = loss / (self.batch_size * self.num_steps)
        self.predicts = predicts
        self.state = prev_results
        self.loss = -tf.reduce_sum(loss)
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def reset_state(self, batch_size):
        result = {}
        for cell in self.cells:
            result[cell.h_t] = np.zeros((batch_size, self.size))
            result[cell.c_t] = np.zeros((batch_size, self.size))
        return result

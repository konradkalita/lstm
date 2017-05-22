import autograd.numpy as np

from tqdm import tqdm

import tensorflow as tf
import time

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from cells import LSTMCell, NASCell

def read_file(file):
  with open(file, encoding='utf-8-sig', mode='U') as f:
    return f.read()


def iterate_dataset(raw_data, batch_size, num_steps):
  batch_length = len(raw_data) // batch_size
  epoch_size = (batch_length - 1) // num_steps
  data = np.zeros([batch_size, batch_length], dtype=np.int32)
  for i in range(batch_size):
      data[i] = raw_data[i * batch_length:(i + 1) * batch_length]
  for i in range(epoch_size):
      x = data[:, i * num_steps:(i+1) * num_steps]
      y = data[:, i * num_steps+1:(i+1) * num_steps+1]
      yield (x, y)

def write_data_to_file(data, file):
  text_file = open(file, "w")
  text_file.write(data)
  text_file.close()

def sigmoid(v):
  return 1 / (1 + np.exp(-v))

# trick kolegi
def onehot(data):
  return np.eye(ALPHABET_SIZE)[data]

def softmax(v):
  exp = np.exp(v)
  exp_sum = np.sum(exp, axis=1)
  return  exp / exp_sum.reshape(exp_sum.size, 1)

def get_desc():
  result = ""
  result += "ALPHABET_SIZE = len(ALPHABET) \n"
  result += "INIT_SCALE = 0.1 \n"
  result += "BATCH_SIZE = 20 \n"
  result += "NUM_STEPS = 20 \n"
  result += "HIDDEN_LAYER_SIZE = 200 \n"
  result += "EPOCHS_NUM = " + str(EPOCHS_NUM) + " \n"
  result += "LAYERS_COUNT = " + str(LAYERS_COUNT) + " \n"
  result += "LEARNING_RATE = " + str(LEARNING_RATE) + " \n"
  result += "DROPOUT = " + str(DROPOUT) + " \n"
  result += "ADAM = " + str(ADAM) + " \n"
  result += "USE_LSTM_CELL = " + str(USE_LSTM_CELL) + " \n"
  return result

train_set = read_file("pan_tadeusz/pan_tadeusz_1_10.txt")
validate_set = read_file("pan_tadeusz/pan_tadeusz_11.txt")
test_set = read_file("pan_tadeusz/pan_tadeusz_12.txt")

ALPHABET = set(train_set).union(set(validate_set)).union(set(test_set))

LETTER_INDX = {}
SIMPLE_LIST = []
for i, l in enumerate(ALPHABET):
  LETTER_INDX[l] = i
  SIMPLE_LIST.append(l)

ALPHABET_SIZE = len(ALPHABET)
INIT_SCALE = 0.1
BATCH_SIZE = 20
NUM_STEPS = 20
HIDDEN_LAYER_SIZE = 200
EPOCHS_NUM = 60
LAYERS_COUNT = 3
LEARNING_RATE = 1.0
DROPOUT = 0.2
USE_LSTM_CELL = True
ADAM = False

logger = get_desc()
desc1 = logger
preplexities = []

timestamp = time.time()
saver_file = "tmp/model" + str(timestamp) + ".ckpt"


def initialize_matrix(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def create_dropout_mask(shape):
  result = tf.to_int64(tf.greater(tf.random_uniform(shape), DROPOUT))
  return tf.cast(result, tf.float32)

class LSTM:
  def __init__(self):
    self.learning_rate = 1.0
    self.build_model()
    self.embedding_nodes()
    self.construct_all()

  def build_model(self):
    if USE_LSTM_CELL:
      self.cells = [LSTMCell(HIDDEN_LAYER_SIZE, ALPHABET_SIZE) for i in range(LAYERS_COUNT)]
    else:
      self.cells = [NASCell(HIDDEN_LAYER_SIZE) for i in range(LAYERS_COUNT)]
    self.apply_dropout = tf.placeholder(tf.bool)
    # self.cells = [LSTM_CELL(), LSTM_CELL(), LSTM_CELL(), LSTM_CELL()]

  def dropout(self, v, mask):
    if DROPOUT == 0.0:
      return v
    else:
      return tf.cond(self.apply_dropout, lambda: (v * mask) / (1 - DROPOUT), lambda: v)
    # return v
    # return (v * mask) / (1 - DROPOUT)

  def embedding_nodes(self):
    self.W_in = initialize_matrix([ALPHABET_SIZE, HIDDEN_LAYER_SIZE])
    self.b_in = initialize_matrix([HIDDEN_LAYER_SIZE])
    self.b_out = initialize_matrix([ALPHABET_SIZE])

  def embedd_input(self, x_t):
    # x_t = tf.expand_dims(x_t, 0)
    return tf.matmul(x_t, self.W_in) + self.b_in

  def embedd_output(self, out):
    W_out = tf.transpose(self.W_in)
    out = tf.matmul(out, W_out) + self.b_out
    return out

  def step(self, x_t, prev_results, cells, masks):
    x_t = self.embedd_input(x_t)

    results = []
    for i in range(len(cells)):
      #vertical dropout
      x_t = self.dropout(x_t, masks[i])
      res = cells[i].step(x_t, prev_results[i][0], prev_results[i][1])
      results.append(res)
      x_t = res[0]
    y = self.embedd_output(x_t)
    y = tf.nn.softmax(y)
    return (y, results)

  def cost_function(self, cells, inputs, labels):
    horizontal_masks = [create_dropout_mask((BATCH_SIZE, HIDDEN_LAYER_SIZE)) for cell in cells]
    vertical_masks = [create_dropout_mask((BATCH_SIZE, HIDDEN_LAYER_SIZE)) for cell in cells]

    prev_results = [[self.dropout(cell.h_t, horizontal_masks[0]), self.dropout(cell.c_t, horizontal_masks[0])] for cell in cells]

    loss = tf.constant([0.0]) # NO IDEA NOW

    predicts = []
    for i in range(BATCH_SIZE):
      #horizontal dropout
      # print("bef", prev_results)
      # print("aft", prev_results)

      y, prev_results = self.step(inputs[i], prev_results, cells, vertical_masks)
      if i == 0: #needed part of graph for generating text
        self.first_memory = prev_results

      for r in range(len(prev_results)):
        prev_results[r][0] = self.dropout(prev_results[r][0], horizontal_masks[r])
        # prev_results[r][1] = self.dropout(prev_results[r][1], horizontal_masks[r])
      # new_results = []
      # for i in range(len(prev_results)):
      #   a = self.dropout(prev_results[i][0], horizontal_masks[i])
      #   b = self.dropout(prev_results[i][1], horizontal_masks[i])
      #   new_results.append((a,b))
      # prev_results = new_results


      predicts.append(y)
      loss += tf.reduce_sum(tf.log(y) * labels[i])

    loss = loss / (BATCH_SIZE * NUM_STEPS)
    self.predicts = predicts
    self.memories = prev_results
    self.loss = -tf.reduce_sum(loss)
    if ADAM:
      self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
    else:
      self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)

  def construct_all(self):
    self.batch_placeholder = tf.placeholder(tf.float32, shape=[None, None, ALPHABET_SIZE])
    self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, None, ALPHABET_SIZE])
    self.cost_function(self.cells, self.batch_placeholder, self.labels_placeholder)

  def reset_memories_dict(self, size):
    result = {}
    for cell in self.cells:
      result[cell.h_t] = np.zeros((size, HIDDEN_LAYER_SIZE))
      result[cell.c_t] = np.zeros((size, HIDDEN_LAYER_SIZE))
    return result

  def data_to_np_onehots(self, data):
    result = []
    for i in range(0, data.shape[1]):
      result.append(onehot(data[:, i]))
    return np.array(result)

  def process_epoch(self, raw_data, do_dropout):
    all_placeholders = self.reset_memories_dict(BATCH_SIZE)
    all_placeholders[self.apply_dropout] = do_dropout

    total_prep = 0.0
    count = 0.0
    for (batch, labels) in tqdm(iterate_dataset(raw_data, BATCH_SIZE, NUM_STEPS), total=(len(raw_data) // BATCH_SIZE - 1) // NUM_STEPS):
      all_placeholders[self.batch_placeholder] = self.data_to_np_onehots(batch)
      all_placeholders[self.labels_placeholder] = self.data_to_np_onehots(labels)
      expected_nodes = [self.train_step, self.loss] + self.memories
      # print(all_placeholders[self.batch_placeholder].shape)
      result = sess.run(expected_nodes, feed_dict=all_placeholders)

      for i in range(len(self.cells)):
        all_placeholders[self.cells[i].h_t] = result[i + 2][0]
        all_placeholders[self.cells[i].c_t] = result[i + 2][1]
      total_prep += result[1]
      count += 1.0
    #   print(np.exp(total_prep / count))
    return np.exp(total_prep / count)

  #dynamic evaluation
  def validate_with_learing(self, raw_data):
    saver = tf.train.Saver()
    saver.save(sess, saver_file)
    preplexity = self.process_epoch(raw_data, False)
    saver.restore(sess, saver_file)
    return preplexity

  def get_letter(self, predicts):
    predicts = predicts.reshape((-1))
    return np.random.choice(ALPHABET_SIZE, p=predicts)

  def generate_text(self, n):
    result = []
    start = "Jam jest Jacek "
    raw_start = list(map(lambda l: LETTER_INDX[l], start))
    placeholders = self.reset_memories_dict(1)
    placeholders[self.apply_dropout] = False

    expected_nodes = [self.predicts[0], self.first_memory]
    for i in raw_start:
      x = np.expand_dims(onehot(i), axis=0)
      x = np.expand_dims(x, axis=0)
      placeholders[self.batch_placeholder] = x
      predicts, memories = sess.run(expected_nodes, feed_dict=placeholders)
      for i in range(len(self.cells)):
        placeholders[self.cells[i].h_t] = memories[i][0]
        placeholders[self.cells[i].c_t] = memories[i][1]

    letter = LETTER_INDX[" "]
    for i in range(n):
      x = np.expand_dims(onehot(letter), axis=0)
      x = np.expand_dims(x, axis=0)
      placeholders[self.batch_placeholder] = x
      predicts, memories = sess.run(expected_nodes, feed_dict=placeholders)
      letter = self.get_letter(predicts)

      for i in range(len(self.cells)):
        placeholders[self.cells[i].h_t] = memories[i][0]
        placeholders[self.cells[i].c_t] = memories[i][1]
      result.append(SIMPLE_LIST[letter])

    return start + ''.join(result)

  def train(self, raw_data, validating_data, test_set):
    global logger
    for i in range(0, EPOCHS_NUM):
      current_epoch = i + 1

      to_print = "Epoch:" + str(current_epoch) + "\n"
      print(to_print)

      train_prep = self.process_epoch(train_set, True)
      validation_prep = self.validate_with_learing(validating_data)
      test_prep = self.validate_with_learing(test_set)
      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      to_print = "Train preplexity: " + str(train_prep) + "\n"
      print(to_print)
      to_print = "Validation preplexity: " + str(validation_prep) + "\n"
      print(to_print)
      preplexities.append(test_prep)
      to_print = "Test preplexity: " + str(test_prep) + "\n"
      print(to_print)
      print("#######################################################")
      print(self.generate_text(300))
      print("#######################################################")

train_set = list(map(lambda l: LETTER_INDX[l], train_set))
validate_set = list(map(lambda l: LETTER_INDX[l], validate_set))
test_set = list(map(lambda l: LETTER_INDX[l], test_set))


sess = tf.InteractiveSession()

def start():
  global sess
  model = LSTM()
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  model.train(train_set, validate_set, test_set)

# start()
# write_data_to_file(logger, "logs/Output" + str(timestamp) + ".txt")


tf.reset_default_graph()

EPOCHS_NUM = 3
LAYERS_COUNT = 2
LEARNING_RATE = 1.0
DROPOUT = 0.2
ADAM = False
USE_LSTM_CELL = True
start()

des = get_desc()
lb = 'Desc_1'
line1, = plt.plot(preplexities, label=lb)
write_data_to_file(des, "ndescs/" + lb + ".txt")
preplexities = []

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig("results")

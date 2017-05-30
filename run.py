#  oświadczam że poniższy kod napisałem samodzielnie - Konrad Kalita

import autograd.numpy as np

import tensorflow as tf
import time
from tqdm import tqdm
from configs import get_configs
import matplotlib.pyplot as plt
from model import LSTMModel


def read_file(file):
    with open(file, encoding='utf-8-sig', mode='U') as f:
        return f.read()


def write_data_to_file(data, file):
    with open(file, "w") as f:
        f.write(data)

PAN_TADEUSZ = True
if PAN_TADEUSZ:
    train_filename = 'pan_tadeusz/pan_tadeusz_1_10.txt'
    valid_filename = 'pan_tadeusz/pan_tadeusz_11.txt'
    test_filename = 'pan_tadeusz/pan_tadeusz_12.txt'
else:
    train_filename = 'ptb/ptb.train.txt'
    valid_filename = 'ptb/ptb.valid.txt'
    test_filename = 'ptb/ptb.test.txt'

def load_data():
    train_set = read_file(train_filename)
    valid_set = read_file(valid_filename)
    test_set = read_file(test_filename)

    def generate_mappers(train_data, valid_data, test_data):
        chars = sorted(list(set(train_data) | set(valid_data) | set(test_data)))
        print(chars)
        c2i = {c: i for i, c in enumerate(chars)}
        i2c = {i: c for i, c in enumerate(chars)}
        return c2i, i2c

    c2i, i2c = generate_mappers(train_set, valid_set, test_set)
    train_data = [c2i[x] for x in train_set]
    valid_data = [c2i[x] for x in valid_set]
    test_data = [c2i[x] for x in test_set]
    return (train_data, valid_data, test_data), (c2i, i2c)


(train_set, valid_set, test_set), (c2i, i2c) = load_data()

ALPHABET = c2i.keys()
ALPHABET_SIZE = len(ALPHABET)

perplexities = []


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


def onehot(data):
    return np.eye(ALPHABET_SIZE)[data]


def to_onehots(data):
    return np.array([onehot(data[:, i]) for i in range(0, data.shape[1])])


def run_epoch(data, is_training, model):
    feed_dict = model.reset_state(BATCH_SIZE)
    feed_dict[model.scope.is_training] = is_training
    total_cost = 0.0
    count = 0.0
    for (batch, labels) in tqdm(iterate_dataset(data, BATCH_SIZE, NUM_STEPS), total=(len(data) // BATCH_SIZE - 1) // NUM_STEPS):
        feed_dict[model.batch] = to_onehots(batch)
        feed_dict[model.labels] = to_onehots(labels)
        expected_nodes = [model.train_step, model.loss] + model.state
        result = sess.run(expected_nodes, feed_dict=feed_dict)

        for i in range(len(model.cells)):
            feed_dict[model.cells[i].h_t] = result[i + 2][0]
            feed_dict[model.cells[i].c_t] = result[i + 2][1]
        total_cost += result[1]
        count += 1.0
    #   print(np.exp(total_cost / count))
    return np.exp(total_cost / count)


def dynamic_eval(data, model):

    timestamp = time.time()
    saver_file = "tmp/model" + str(timestamp) + ".ckpt"

    saver = tf.train.Saver()
    saver.save(sess, saver_file)
    perplexity = run_epoch(data, False, model)
    saver.restore(sess, saver_file)
    return perplexity


def get_letter(predicts):
    predicts = predicts.reshape((-1))
    return np.random.choice(ALPHABET_SIZE, p=predicts)


def generate_text(n, model):
    result = []
    start = ''
    if PAN_TADEUSZ:
        start = "Jam jest Jacek"
    else :
        start = "no it was n't black monday"
    raw_start = list(map(lambda l: c2i[l], start))
    placeholders = model.reset_state(1)
    placeholders[model.scope.is_training] = False

    expected_nodes = [model.predicts[0], model.first_memory]
    for i in raw_start:
        x = np.expand_dims(onehot(i), axis=0)
        x = np.expand_dims(x, axis=0)
        placeholders[model.batch] = x
        predicts, state = sess.run(expected_nodes, feed_dict=placeholders)
        for i in range(len(model.cells)):
            placeholders[model.cells[i].h_t] = state[i][0]
            placeholders[model.cells[i].c_t] = state[i][1]

    letter = c2i[" "]
    for i in range(n):
        x = np.expand_dims(onehot(letter), axis=0)
        x = np.expand_dims(x, axis=0)
        placeholders[model.batch] = x
        predicts, state = sess.run(expected_nodes, feed_dict=placeholders)
        letter = get_letter(predicts)

        for i in range(len(model.cells)):
            placeholders[model.cells[i].h_t] = state[i][0]
            placeholders[model.cells[i].c_t] = state[i][1]
        result.append(i2c[letter])

    return start + ''.join(result)


def train(train_set, valid_set, test_set, model):
    perplexities = []
    for i in range(0, EPOCHS_NUM):


        to_print = "Epoch: %d\n" % (i+1)
        print(to_print)

        train_perplexity = run_epoch(train_set, True, model)
        test_perplexity = dynamic_eval(test_set, model)
        print("----------------------------------------------------")
        to_print = "Train score: %s\n" % (train_perplexity)
        print(to_print)
        # to_print = "Validation score: %s\n" % (validation_prep)
        # print(to_print)
        perplexities.append(test_perplexity)
        to_print = "Test score: %s\n" % (test_perplexity)
        print(to_print)
        print("----------------------------------------------------")
        print(generate_text(300, model))
        print("----------------------------------------------------")
    return perplexities


def start(config):
    global sess
    print(config)
    model = LSTMModel(config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    return train(train_set, valid_set, test_set, model)


EPOCHS_NUM = 20
BATCH_SIZE = 20
NUM_STEPS = 20

fig, ax = plt.subplots()
ax.set_yticks(np.arange(0, 10, 0.5))
ax.grid(which='both')
sess = tf.InteractiveSession()
configs = get_configs(ALPHABET_SIZE)
series = []
xs = list(range(1, EPOCHS_NUM+1))
for label, config in configs:
    tf.reset_default_graph()
    perplexities = start(config)
    ax.plot(xs, perplexities, label=label)
    series.append(perplexities)

legend = ax.legend(loc='upper center', shadow=True)

plt.show()

plt.savefig("results")

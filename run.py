import autograd.numpy as np

import tensorflow as tf
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from model import LSTM


def read_file(file):
    with open(file, encoding='utf-8-sig', mode='U') as f:
        return f.read()


def write_data_to_file(data, file):
    with open(file, "w") as f:
        f.write(data)


def load_data():
    train_set = read_file("pan_tadeusz/pan_tadeusz_1_10.txt")
    valid_set = read_file("pan_tadeusz/pan_tadeusz_11.txt")
    test_set = read_file("pan_tadeusz/pan_tadeusz_12.txt")

    def generate_mappers(train_data, valid_data, test_data):
        chars = sorted(list(set(train_data) | set(valid_data) | set(test_data)))
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

LETTER_INDX = c2i
SIMPLE_LIST = i2c

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


def data_to_np_onehots(data):
    result = []
    for i in range(0, data.shape[1]):
        result.append(onehot(data[:, i]))
    return np.array(result)


def process_epoch(raw_data, do_dropout, model):
    all_placeholders = model.reset_memories_dict(BATCH_SIZE)
    all_placeholders[model.apply_dropout] = do_dropout

    total_prep = 0.0
    count = 0.0
    for (batch, labels) in tqdm(iterate_dataset(raw_data, BATCH_SIZE, NUM_STEPS), total=(len(raw_data) // BATCH_SIZE - 1) // NUM_STEPS):
        all_placeholders[model.batch_placeholder] = data_to_np_onehots(batch)
        all_placeholders[model.labels_placeholder] = data_to_np_onehots(labels)
        expected_nodes = [model.train_step, model.loss] + model.memories
        result = sess.run(expected_nodes, feed_dict=all_placeholders)

        for i in range(len(model.cells)):
            all_placeholders[model.cells[i].h_t] = result[i + 2][0]
            all_placeholders[model.cells[i].c_t] = result[i + 2][1]
        total_prep += result[1]
        count += 1.0
    #   print(np.exp(total_prep / count))
    return np.exp(total_prep / count)


timestamp = time.time()
saver_file = "tmp/model" + str(timestamp) + ".ckpt"


def validate_with_learing(raw_data, model):
    saver = tf.train.Saver()
    saver.save(sess, saver_file)
    preplexity = process_epoch(raw_data, False, model)
    saver.restore(sess, saver_file)
    return preplexity


def get_letter(predicts):
    predicts = predicts.reshape((-1))
    return np.random.choice(ALPHABET_SIZE, p=predicts)


def generate_text(n, model):
    result = []
    start = "Jam jest Jacek "
    raw_start = list(map(lambda l: LETTER_INDX[l], start))
    placeholders = model.reset_memories_dict(1)
    placeholders[model.apply_dropout] = False

    expected_nodes = [model.predicts[0], model.first_memory]
    for i in raw_start:
        x = np.expand_dims(onehot(i), axis=0)
        x = np.expand_dims(x, axis=0)
        placeholders[model.batch_placeholder] = x
        predicts, memories = sess.run(expected_nodes, feed_dict=placeholders)
        for i in range(len(model.cells)):
            placeholders[model.cells[i].h_t] = memories[i][0]
            placeholders[model.cells[i].c_t] = memories[i][1]

    letter = LETTER_INDX[" "]
    for i in range(n):
        x = np.expand_dims(onehot(letter), axis=0)
        x = np.expand_dims(x, axis=0)
        placeholders[model.batch_placeholder] = x
        predicts, memories = sess.run(expected_nodes, feed_dict=placeholders)
        letter = get_letter(predicts)

        for i in range(len(model.cells)):
            placeholders[model.cells[i].h_t] = memories[i][0]
            placeholders[model.cells[i].c_t] = memories[i][1]
        result.append(SIMPLE_LIST[letter])

    return start + ''.join(result)


def train(train_set, validating_data, test_set, model):
    global logger
    perplexities = []
    for i in range(0, EPOCHS_NUM):
        current_epoch = i + 1

        to_print = "Epoch:" + str(current_epoch) + "\n"
        print(to_print)

        train_prep = process_epoch(train_set, True, model)
        validation_prep = validate_with_learing(validating_data, model)
        test_prep = validate_with_learing(test_set, model)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Evaluation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        to_print = "Train preplexity: " + str(train_prep) + "\n"
        print(to_print)
        to_print = "Validation preplexity: " + str(validation_prep) + "\n"
        print(to_print)
        perplexities.append(test_prep)
        to_print = "Test preplexity: " + str(test_prep) + "\n"
        print(to_print)
        print("#######################################################")
        print(generate_text(300, model))
        print("#######################################################")
    return perplexities


sess = tf.InteractiveSession()


def start():
    global sess
    model = LSTM(200, ALPHABET_SIZE, BATCH_SIZE, LAYERS_COUNT, DROPOUT, USE_LSTM_CELL, NUM_STEPS, LEARNING_RATE)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(LAYERS_COUNT, HIDDEN_LAYER_SIZE)
    return train(train_set, valid_set, test_set, model)


tf.reset_default_graph()

EPOCHS_NUM = 3
LAYERS_COUNT = 1
LEARNING_RATE = 1.0
DROPOUT = 0.2
ADAM = False
USE_LSTM_CELL = True
perplexities = start()


lb = 'Desc_1'
line1, = plt.plot(perplexities, label=lb)
# write_data_to_file(des, "ndescs/" + lb + ".txt")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.savefig("results")

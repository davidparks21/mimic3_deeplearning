#!/soe/davidparks21/anaconda2/bin/python
import tensorflow as tf
import numpy as np
import argparse, re

tensor_regex = re.compile('.*:\d*')


def build_model(hyperparameters):
    learning_rate = hyperparameters['learning_rate']

    n_input = 28  # MNIST data input (img shape: 28*28)
    n_steps = 28  # timesteps
    n_hidden = 128  # hidden layer num of features
    n_classes = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    # Define weights and biases
    w1 = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='w1')
    b1 = tf.Variable(tf.random_normal([n_classes]), name='w2')

    # Define a lstm cell with tensorflow
    lstm_cell = tf.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, name='lstm_cell')

    # Get lstm cell output
    outputs, states = tf.rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    pred = tf.add(tf.matmul(outputs[-1], w1), b1, name='pred')

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='optimizer')

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Initializing the variables
    init = tf.initialize_all_variables(name='init')


def train_ann(hyperparameters, dataset):
    with tf.Session() as sess:
        build_model(hyperparameters)
        sess.run(t('init'))

        batch_x, batch_y = dataset.get_batch(hyperparameters['batch_size'])
        sess.run(t('optimizer'), feed_dic={t('x'):batch_x, t('y'):batch_y})



# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.get_default_graph().get_tensor_by_name(tensor_name)


def prediction():
    None #TODO


class DataSet:
    None #TODO



#################################################
## Command line execution for training network
#################################################
if __name__ == '__main__':
    DEFAULT_TRAINING_ITERS = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_file', default="../models/localize_model", required=False, help='Name of the checkpoint file to save (without file extension)')
    parser.add_argument('-x', '--datafile_features', default="../data/features.npy", required=False, help='Path and file name of numpy data file of features')
    parser.add_argument('-y', '--datafile_labels', default="../data/labels.npy", required=False, help='Path and file name of numpy data file of labels')
    parser.add_argument('-l', '--loadmodel', default=None, required=False, help='Specify a model name to load')
    parser.add_argument('-i', '--iterations', default=DEFAULT_TRAINING_ITERS, required=False, help='Number of training iterations')
    args = vars(parser.parse_args())

    # Hyperparameters is a dictionary of each hyper parameter along with an array where the 0th element
    # is the best value, and the 1...n elements are options to search during hyperparameter optimization
    hyperparameters = {
        'iterations':       [args['iterations']],
        'learning_rate':    [0.0010,    0.0001, 0.0010, 0.0100],
        'batch_size ':      [128,       32, 64, 128, 192, 256],
    }

    train_ann(hyperparameters, DataSet(args['datafile_features'], args['datafile_labels']))
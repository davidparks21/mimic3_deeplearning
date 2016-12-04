#!/soe/davidparks21/anaconda2/bin/python
import tensorflow as tf
import numpy as np
import argparse, re, random
from DataSet import DataSet as Dataset
from collections import namedtuple

tensor_regex = re.compile('.*:\d*')


def build_model(hyperparameters):
    learning_rate = hyperparameters['learning_rate']
    m = namedtuple('ModelObjects', '')   # container for the tensors and ops we'll need to use elsewhere with convenient object-like syntax

    n_input = 8  # MNIST data input (img shape: 28*28)
    n_steps = 10  # max lenght of sequence
    n_hidden = 128  # hidden layer num of features
    n_classes = 3  # MNIST total classes (0-9 digits)

    # A vector of sequence lengths
    m.X_lengths = tf.placeholder(tf.float32, [hyperparameters['batch_size']])

    # tf Graph input
    m.X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X')
    m.y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    # Define weights and biases
    w1 = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='w1')
    b1 = tf.Variable(tf.random_normal([n_classes]), name='w2')

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    m.outputs, m.states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        dtype=tf.float32,
        sequence_length=m.X_lengths,
        inputs=m.X)

    # Linear activation, using rnn inner loop last output
    val = tf.transpose(m.outputs, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)          # Get just the last prediction from the RNN
    m.pred = tf.add(tf.matmul(last, w1), b1, name='pred')

    # Define loss and optimizer
    m.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(m.pred, m.y), name='cost')
    m.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(m.cost, name='optimizer')

    # Evaluate model
    m.correct_pred = tf.equal(tf.argmax(m.pred, 1), tf.argmax(m.y, 1), name='correct_pred')
    m.accuracy = tf.reduce_mean(tf.cast(m.correct_pred, tf.float32), name='accuracy')

    # Initializing the variables
    m.init = tf.initialize_all_variables()

    return m

def train_ann(hyperparameters, dataset):
    with tf.Session() as sess:
        m = build_model(hyperparameters)
        sess.run(m.init)

        batch_x, batch_X_lengths, batch_y = dataset.next_batch(hyperparameters['batch_size'])
        feed ={m.X:batch_x, m.y:batch_y, m.X_lengths:batch_X_lengths}
        for i in range(1,10):
            result = sess.run([m.optimizer, m.cost], feed_dict=feed)
            print result[1]

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
    iteration_num = 0
    RUN_SINGLE_ITERATION = True

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
        'learning_rate':    0.0010, #[0.0010,    0.0001, 0.0010, 0.0100],
        'batch_size':       2,  #[2]#,       32, 64, 128, 192, 256],
    }

    test_max_length = 10
    test_n_features = 5

    X = np.random.randn(2, 10, 8)   # Batch=2, max_sequence_length=9 (first entry saved for sequence length), num_features=8
    X[1,6:] = 0                     # set 2nd sample to be length 5 with 0 padding after that
    tmp_lengths = np.array([10,6])
    tmp_labels = np.array([[0,1,1],[0,1,1]])

    # train_ann(hyperparameters, DataSet(args['datafile_features'], args['datafile_labels']))
    dataset = Dataset(sample_sequence_features=X, sequence_lengths=tmp_lengths, labels=tmp_labels)
    train_ann(hyperparameters, dataset)

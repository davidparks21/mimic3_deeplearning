import numpy as np

class DataSet:
    # Input samples_sequence_features & labels  _OR_  input the data files to initialize
    def __init__(self, sample_sequence_features=None, sequence_lengths=None, labels=None,
                 datafile_sample_sequence_features=None, datafile_labels=None):
        if sample_sequence_features is not None:
            # A 0-padded numpy matrix of size: [num_samples X max_sequence_length X num_features]
            assert sample_sequence_features.shape[0] == sequence_lengths.shape[0] == labels.shape[0]
            assert np.all(sequence_lengths <= sample_sequence_features.shape[1])

            self.sample_sequence_features = sample_sequence_features
            self.sequence_lengths = sequence_lengths
            self.labels = labels
        else:
            None #TODO - implement later

        self._samples_consumed = 0
        self._ix_permutation = np.random.permutation(np.shape(self.labels)[0])

    def next_batch(self, batch_size):
        batch_ix = self._ix_permutation[0:batch_size]
        self._ix_permutation = np.roll(self._ix_permutation, batch_size)

        # keep track of how many samples have been consumed and reshuffle after an epoch has elapsed
        self._samples_consumed += batch_size
        if self._samples_consumed > np.shape(self.labels)[0]:
            self._ix_permutation = np.random.permutation(np.shape(self.labels)[0])
            self._samples_consumed = 0

        return self.sample_sequence_features[batch_ix, :, :], self.sequence_lengths[batch_ix], self.labels[batch_ix]

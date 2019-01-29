#!/usr/bin/env python3
'''
Module containing data augmentation techniques
'''

import numpy as np


class SignalShift(object):
    '''
    Shift the signal 
    '''

    def __init__(self, data_len=-4, shift_len=10, dim=-1):
        self.data_len = data_len
        self.shift_len = shift_len
        self.dim = dim

    def __call__(self, X):
        num_shits = int(X.shape[-1]/self.shift_len)

        def _augment_and_duplicate_labels(sample):
            if len(sample.shape) == 2:
                raw_data, labels = sample[:, :self.data_len], sample[:, self.data_len:]
            else:
                raw_data, labels = sample[:self.data_len], sample[self.data_len:]
            new_data = np.roll(raw_data, shift=self.shift_len, axis=self.dim)
            return np.hstack((new_data, labels))
        new_data = np.vstack([_augment_and_duplicate_labels(X) for _ in range(num_shits)])
        return new_data


class Invert(object):
    '''
    Invert the signal
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return X * -1

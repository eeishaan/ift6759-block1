#!/usr/bin/env python3
'''
Module containing data transformation
'''

import torch


class ToTensor(object):
    '''
    Transformer for converting numpy array to a tensor object
    '''

    def __init__(self):
        pass
    
    def __call__(self, X):
        return torch.from_numpy(X)


class ToNumpy(object):
    '''
    Transformer for converting tensor object to a numpy object
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return X.numpy()
    
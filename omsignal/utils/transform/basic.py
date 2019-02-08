#!/usr/bin/env python3
'''
Module with basic tranformations
'''
import json

import numpy as np
import torch

from omsignal.utils.transform import OmTransform


class ClipAndFlatten(OmTransform):
    '''
    Remove unnecessary rows and flatten across batches.
    This is useful when segmentation is used as a transformer.
    As the number of rows is not fixed, the segmenter sends a tensor
    of shape (200, segment size). Not all the rows are relevant.
    Rows which have label -1 are irrelevant. This transformer
    removes irrelevant rows and construct a flattened batch.
    It also preserves the boundary information.
    '''

    def __init__(self, segment_size, label_size=4):
        self.segment_size = segment_size
        self.label_size = label_size

    def __call__(self, x, y):
        assert len(x) == len(y)
        boundaries = []
        last_boundary = 0
        res_x, res_y = \
            x.new_empty((1, self.segment_size)), y.new_empty(
                (1, self.label_size))
        for batch in range(len(y)):
            relevant_rows = int((y[batch, :, 3].ge(0) == 0).nonzero()[0])
            new_boundary = last_boundary + relevant_rows
            last_boundary = new_boundary
            boundaries.append(new_boundary)
            res_x = torch.cat((res_x, x[batch, :relevant_rows, :]))
            res_y = torch.cat((res_y, y[batch, :relevant_rows, :]))

        return res_x[1:, :], res_y[1:, :], boundaries


class LabelSeparator(OmTransform):
    '''
    Separate out raw data from labels
    '''

    def __init__(self, label_len=4):
        self.label_len = label_len

    def __call__(self, x):
        return x[:-1*self.label_len], x[-1*self.label_len:]


class RemapLabels(OmTransform):
    '''
    Remap ids to a continuous interval
    '''

    def __init__(self, file_path):
        self.file_path = file_path
        self.map = {}
        self.idx = -1

    def __call__(self, X):
        X = np.copy(X)
        label = str(int(X[-1]))
        if label not in self.map:
            self.idx += 1
            self.map[label] = self.idx
        X[-1] = self.map[label]
        return X

    def state(self):
        return self.map

    def save(self):
        with open(self.file_path, 'w') as fob:
            json.dump(self.map, fob)


class ReverseLabelMap(OmTransform):
    '''
    Transformer to remap ids to true ids
    '''

    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as fob:
            label_map = json.load(fob)
        self.map = {v: k for k, v in label_map.items()}

    def __call__(self, x):
        return self.map[x]


class ToNumpy(OmTransform):
    '''
    Transformer for converting tensor object to a numpy object
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return X.numpy()


class ToTensor(OmTransform):
    '''
    Transformer for converting numpy array to a tensor object
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return torch.from_numpy(X)

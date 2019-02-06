#!/usr/bin/env python3
from pathlib import Path

from omsignal.utils.memfile import read_memfile


def evaluate(dataset_file, model):
    # read the dataset file
    print('hello')
    exit()
    dataset_file = Path(dataset_file)
    test_data = read_memfile(dataset_file, shape=(160, 3750), dtype='float32')

    # process the dataset

    # get the predictions

    # remap labels
    # return predictions

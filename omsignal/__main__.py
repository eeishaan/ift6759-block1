#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from omsignal.constants import (RESULT_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.utils.loader import get_vector_and_labels


def task1(train_vectors, train_labels, output_folder):
    '''
    Pick three ids at random and plot their line graph
    '''
    size=3

    all_indices = np.arange(32)
    np.random.shuffle(all_indices)
    random_ids = all_indices[:size]
    random_window_indices = np.random.randint(5, size=size)
    random_indices = np.multiply(random_ids, random_window_indices)
    subset = np.take(train_vectors, random_indices, axis=0)
    label_subset = np.take(train_labels, random_indices, axis=0)
    
    for i in range(size):
        plt.figure(i+1)
        plt.title("User ID: {}".format(int(label_subset[i][-1])))
        plt.plot(subset[i])
        plt.savefig(output_folder / 'vis{}.png'.format(i+1))


def main():
    '''
    Main function
    '''

    # fix seed for reproducability
    np.random.seed(1)

    # turn off interactive plotting
    plt.ioff()

    # separate out the labels and raw data
    train_vectors, train_labels = get_vector_and_labels(TRAIN_LABELED_FILE)
    valid_vectors, valid_labels = get_vector_and_labels(VALIDATION_LABELED_FILE)

    task1_output_folder = RESULT_DIR / 'task1' 
    task1(train_vectors, train_labels, task1_output_folder)


if __name__ == '__main__':
    main()

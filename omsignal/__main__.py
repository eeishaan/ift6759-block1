#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from omsignal.constants import (RESULT_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.utils.dim_reduction import SVDTransform, TSNETransform
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


def dim_reduction_task(train_vectors, train_labels, out_dir):
    '''
    Reduce dimension and plot scatter plots
    '''
    out_dim = 2
    

    def _reduce_and_plot(X, labels, transformer):
        X = transformer(X)
        color_dict = {}
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for x, label in zip(X, labels):
            user_id = label[-1]
            # use this to prevent multiple legend entries
            arg_dict = {}
            if user_id not in color_dict:
                arg_dict['label'] = user_id
                color_dict[user_id] = np.random.rand(1,3)
            arg_dict['c'] = color_dict[user_id]
            ax.scatter(x[0], x[1], **arg_dict)
        
        plt.title("Scatter plot for {}".format(transformer.name))
        plt.legend(loc=2)
        plt.savefig(out_dir / "{}.png".format(transformer.name))
        
        
    # initialize reduction transformer
    svd_transform = SVDTransform(out_dim)
    tsne_transform = TSNETransform(out_dim)

    # transform using SVD
    _reduce_and_plot(train_vectors, train_labels, svd_transform)

    # transform using t-SNE 
    _reduce_and_plot(train_vectors, train_labels, tsne_transform)
    


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
    os.makedirs(task1_output_folder, exist_ok=True)
    task1(train_vectors, train_labels, task1_output_folder)

    scatter_plot_dir = RESULT_DIR / 'dim_reduction'
    os.makedirs(scatter_plot_dir, exist_ok=True)
    dim_reduction_task(train_vectors, train_labels, scatter_plot_dir)


if __name__ == '__main__':
    main()

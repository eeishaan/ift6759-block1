#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(file_name, labels, predicted):
    c_matrix = confusion_matrix(labels, predicted)
    plt.figure()
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=10)
    plt.imshow(c_matrix)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(file_name)

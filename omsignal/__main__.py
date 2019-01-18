#!/usr/bin/env python3

import os
import yaml

from omsignal.loader import get_vector_and_labels


def set_config_globals():
    file_loc = os.path.dirname(os.path.abspath(__file__))
    constants_file = os.path.join(file_loc, 'constants.yml')
    with open(constants_file, 'r') as fob:
        constants = yaml.load(fob)
    for key, value in constants.items():
        globals()[key] = value


def main():
    '''
    Main function
    '''

    # set global configuration variables
    set_config_globals()

    train_labeled_data_file = os.path.join(ROOT_DIR, TRAIN_LABELED_FILE)
    validation_labeled_data_file = os.path.join(ROOT_DIR, VALIDATION_LABELED_FILE)

    # separate out the labels and raw data
    train_vectors, train_labels = get_vector_and_labels(train_labeled_data_file)
    valid_vectors, valid_labels = get_vector_and_labels(validation_labeled_data_file)


if __name__ == '__main__':
    main()

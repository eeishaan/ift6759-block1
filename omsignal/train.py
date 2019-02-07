#!/usr/bin/env python3
import argparse

from omsignal.runner import (run_cnn_classification, run_cnn_multi_task,
                             run_cnn_regression)

MODEL_EXP_MAP = {
    'cnn_classification': run_cnn_classification,
    'cnn_regression': run_cnn_regression,
    'cnn_multi_task': run_cnn_multi_task,
    'best_model': run_cnn_multi_task,
}


def get_train_parser(parent=None):
    '''
    Return command line arguments
    '''
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('train', help='Test pre-trained models')

    parser.add_argument(
        '--model',
        type=str,
        help='Model type to train',
        choices=MODEL_EXP_MAP.keys(),
        required=True,
    )

    parser.add_argument(
        '--test-data',
        type=str,
        help='Test data file location',
        required=True,
    )

    parser.add_argument(
        '--train-data',
        type=str,
        help='Train data file location',
        required=True,
    )

    parser.add_argument(
        '--params',
        type=str,
        help='Model param file location. '
        'For information about param file format refer README.md'
    )

    return parser

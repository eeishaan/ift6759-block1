#!/usr/bin/env python3

import argparse
import logging
import os
import sys

from omsignal.constants import DATA_ROOT_DIR, MODEL_DIR
from omsignal.train import MODEL_EXP_MAP
from omsignal.utils.misc import check_file


def get_test_parser(parent=None):
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser('test', help='Test pre-trained models')

    parser.add_argument(
        '--model',
        type=str,
        help='Type of model to evaluate',
        choices=MODEL_EXP_MAP.keys(),
        required=True,
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='File path of the model',
        required=True
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='File path of the test data',
        required=True
    )

    return parser


def run_eval(args):
    data_file = args.data_file
    model_file = args.model_file
    model_file = check_file(model_file, MODEL_DIR)
    if model_file is None:
        return
    data_file = check_file(data_file, DATA_ROOT_DIR)
    if data_file is None:
        return
    # exp = MODEL_EXP_MAP[args.map]


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args()
    run_eval(args)

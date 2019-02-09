#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

from omsignal.constants import DATA_ROOT_DIR, ID_MAPPING, MODEL_DIR, PARAM_DIR
from omsignal.experiments.cnn_experiment import (MultiTaskExperiment,
                                                 RegressionNetEperiment,
                                                 SimpleNetExperiment)
from omsignal.experiments.deterministic import DeterministicExp
from omsignal.utils.loader import get_test_dataloader
from omsignal.utils.memfile import read_memfile
from omsignal.utils.misc import check_file
from omsignal.utils.transform.basic import ReverseLabelMap
from omsignal.utils.transform.preprocessor import Preprocessor, SignalSegmenter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_test_parser(parent=None):
    '''
    Construct argparser for test script
    '''

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


def test_cnn_classification(model_file, test_data_file):
    # segment = hearbeat
    segmenter = SignalSegmenter()

    # get loader for segmented preprocessed data
    test_loader, row_mapping = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    # mapper for mapping the id to true id
    rev_id_mapper = ReverseLabelMap(ID_MAPPING)

    # instantiate an experiment
    exp_cls = SimpleNetExperiment(
        model_file,
        device=device,
    )

    # load the experiment from file
    exp_cls.load_experiment()

    # run the exp on test data
    preds = exp_cls.test(test_loader)
    preds = np.array(preds, dtype='int')

    # take a majority vote among segments
    y_pred_majority = np.array([])
    for i in range(160):
        index_of_int = row_mapping == i
        counts = np.bincount(preds[index_of_int].astype(int))
        y_pred_majority = np.append(y_pred_majority, np.argmax(counts))

    # map the label to true labels
    y_pred_majority = np.array([rev_id_mapper(i) for i in y_pred_majority])
    return y_pred_majority.astype('float32')


def test_cnn_regression(model_file, test_data_file):
    # segment = heart beat
    segmenter = SignalSegmenter(take_average=True)

    # get preprocessed segmented data
    test_loader, _ = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    # initialize experiment
    exp_cls = RegressionNetEperiment(
        model_file,
        device=device,
    )
    # load the experiment from file
    exp_cls.load_experiment()

    # run on test data
    preds = exp_cls.test(test_loader)

    return preds.astype('float32')


def test_cnn_multi_task(model_file, test_data_file):
    # segment = heart beat
    segmenter = SignalSegmenter(take_average=True)

    # get preprocessed segmented data
    test_loader, _ = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    # reverse mapper for converting id to true id
    rev_id_mapper = ReverseLabelMap(ID_MAPPING)

    # initialize experiment
    exp_cls = RegressionNetEperiment(
        model_file,
        device=device,
    )

    # load the experiment from file
    exp_cls.load_experiment()

    # run on test data
    preds = exp_cls.test(test_loader)

    # remap to true id
    preds[:, -1] = np.array([rev_id_mapper(p) for p in preds[:, -1]])
    return preds.astype('float32')


def test_deterministic(model_file, test_data_file):
    # load the data
    test_data = read_memfile(
        test_data_file, shape=(160, 3750), dtype='float32')

    # pre-process data
    preprocessor = Preprocessor()
    test_data = torch.tensor(test_data)
    test_data = preprocessor(test_data)

    # mapper to map to true id
    rev_id_mapper = ReverseLabelMap(ID_MAPPING)

    # load exp from file
    exp_cls = DeterministicExp.load_experiment(model_file)

    # run on test data
    preds = exp_cls.test(test_data)

    # remap ids to true ids
    preds[:, -1] = np.array([rev_id_mapper(p) for p in preds[:, -1]])
    return preds.astype('float32')


MODEL_EXP_MAP = {
    'cnn_classification': {
        'test_func': test_cnn_classification,
        'param_file': PARAM_DIR / 'cnn_classification.yml',
    },
    'cnn_regression': {
        'test_func': test_cnn_regression,
        'param_file': PARAM_DIR / 'cnn_regression.yml',
    },
    'cnn_multi_task': {
        'test_func': test_cnn_multi_task,
        'param_file': PARAM_DIR / 'cnn_multi_task.yml',
    },
    'deterministic_task': {
        'test_func': test_deterministic,
        'param_file': PARAM_DIR / 'deterministic.yml',
    },
    'best_model': {
        'test_func': test_deterministic,
        'param_file': PARAM_DIR / 'deterministic.yml',
    },
}


def test(args):
    data_file = args.data_file
    model_file = args.model_path
    model_file = check_file(model_file, MODEL_DIR)
    if model_file is None:
        exit(1)
    data_file = check_file(data_file, DATA_ROOT_DIR)
    if data_file is None:
        exit(1)
    model = args.model

    # find the relevant test function
    test_func = MODEL_EXP_MAP[model]['test_func']

    # if no model file is specified, use the one specified in params file
    if model_file is None:
        param_file = MODEL_EXP_MAP[model]['param_file']
        param_file = check_file(param_file, PARAM_DIR)
        if param_file is None:
            exit(1)
        with open(param_file) as fob:
            params = yaml.load(fob)
        model_file = params['model_file']
    return test_func(model_file, data_file)


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args()
    print(test(args))

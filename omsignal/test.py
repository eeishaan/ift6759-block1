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
from omsignal.utils.loader import get_test_dataloader, get_vector_and_labels
from omsignal.utils.memfile import read_memfile
from omsignal.utils.misc import check_file
from omsignal.utils.transform.basic import ReverseLabelMap
from omsignal.utils.transform.preprocessor import Preprocessor, SignalSegmenter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def test_cnn_classification(model_file, test_data_file):
    segmenter = SignalSegmenter()
    test_loader, row_mapping = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    rev_id_mapper = ReverseLabelMap(ID_MAPPING)
    exp_cls = SimpleNetExperiment(
        model_file,
        device=device,
    )
    exp_cls.load_experiment()
    preds = exp_cls.test(test_loader)
    preds = np.array(preds, dtype='int')
    # vote and remap
    y_pred_majority = np.array([])
    for i in range(160):
        index_of_int = row_mapping == i
        counts = np.bincount(preds[index_of_int].astype(int))
        y_pred_majority = np.append(y_pred_majority, np.argmax(counts))
    y_pred_majority = np.array([rev_id_mapper(i) for i in y_pred_majority])
    return y_pred_majority


def test_cnn_regression(model_file, test_data_file):
    segmenter = SignalSegmenter(take_average=True)
    test_loader, _ = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    exp_cls = RegressionNetEperiment(
        model_file,
        device=device,
    )
    exp_cls.load_experiment()
    preds = exp_cls.test(test_loader)
    # vote and remap
    return preds


def test_cnn_multi_task(model_file, test_data_file):
    segmenter = SignalSegmenter(take_average=True)
    test_loader, _ = get_test_dataloader(test_data_file, segmenter)
    if test_loader is None:
        exit(1)

    rev_id_mapper = ReverseLabelMap(ID_MAPPING)
    exp_cls = RegressionNetEperiment(
        model_file,
        device=device,
    )
    exp_cls.load_experiment()
    preds = exp_cls.test(test_loader)
    preds[:, -1] = np.array([rev_id_mapper(p) for p in preds[:, -1]])
    return preds


def test_deterministic(model_file, test_data_file):
    test_data, _ = get_vector_and_labels(test_data_file)
    preprocessor = Preprocessor()
    test_data = torch.tensor(test_data)
    test_data = preprocessor(test_data)

    rev_id_mapper = ReverseLabelMap(ID_MAPPING)
    exp_cls = DeterministicExp.load_experiment(model_file)
    preds = exp_cls.test(test_data)
    preds[:, -1] = np.array([rev_id_mapper(p) for p in preds[:, -1]])
    return preds


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

    test_func = MODEL_EXP_MAP[model]['test_func']
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

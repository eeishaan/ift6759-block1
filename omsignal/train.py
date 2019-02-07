#!/usr/bin/env python3
import argparse
import os

import torch

from omsignal.constants import (MODEL_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import (MultiTaskExperiment,
                                                 RegressionNetEperiment,
                                                 SimpleNetExperiment)
from omsignal.experiments.lstm_experiment import LSTMExperiment
from omsignal.utils.loader import get_dataloader
from omsignal.utils.transform.basic import RemapLabels
from omsignal.utils.transform.preprocessor import (LSTMSegmenter, Preprocessor,
                                                   SignalSegmenter,
                                                   get_preprocessed_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_cnn_classification():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "simple_net.pt"

    # create id remap transformer
    remap = RemapLabels()

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        TRAIN_LABELED_FILE,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=128
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        VALIDATION_LABELED_FILE,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=SignalSegmenter(),
        shuffle=False,
        batch_size=128
    )

    simplenet_exp = SimpleNetExperiment(
        model_file,
        optimiser_params={
            'lr': 0.1
        },
        device=device
    )
    print('started training')
    simplenet_exp.train(
        train_loader,
        epochs=3000,
        validation_dataloader=valid_loader)


def train_lstm_exp():
    """
    Main function
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "lstm.pt"

    # remap labels
    remap = RemapLabels()

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        TRAIN_LABELED_FILE,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=LSTMSegmenter(),
        shuffle=True,
        batch_size=8
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        VALIDATION_LABELED_FILE,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=LSTMSegmenter(),
        shuffle=False,
        batch_size=8
    )

    lstm_exp = LSTMExperiment(
        model_file,
        optimiser_params={
            'lr': 0.0005,
            'weight_decay': 0.0001
        },
        model_params={
            'device': device,
            'n_layers': 1,
        },
        device=device
    )
    print('started training')
    lstm_exp.train(
        train_loader,
        epochs=120,
        validation_dataloader=valid_loader)


def train_cnn_regression():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "regression_net.pt"

    # remap labels
    remap = RemapLabels()

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        TRAIN_LABELED_FILE,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=160
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        VALIDATION_LABELED_FILE,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=160
    )

    regnet_exp = RegressionNetEperiment(
        model_file,
        optimiser_params={
            'lr': 0.01
        },
        device=device
    )
    print('started training')
    regnet_exp.train(
        train_loader,
        epochs=4000,
        validation_dataloader=valid_loader)


def train_cnn_multi_task():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "multi_task_net.pt"

    # remap labels
    remap = RemapLabels()

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        TRAIN_LABELED_FILE,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=160
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        VALIDATION_LABELED_FILE,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=160
    )

    regnet_exp = MultiTaskExperiment(
        model_file,
        optimiser_params={
            'lr': 0.01
        },
        device=device
    )
    print('started training')
    regnet_exp.train(
        train_loader,
        epochs=4000,
        validation_dataloader=valid_loader)


# need to define below function definitions
MODEL_EXP_MAP = {
    'cnn_classification': train_cnn_classification,
    'cnn_regression': train_cnn_regression,
    'cnn_multi_task': train_cnn_multi_task,
    'best_model': train_cnn_multi_task,
}

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()

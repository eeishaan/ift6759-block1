#!/usr/bin/env python3
import argparse
import os

import torch
import yaml

from omsignal.constants import (ID_MAPPING, MODEL_DIR, PARAM_DIR,
                                TRAIN_LABELED_FILE, VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import (MultiTaskExperiment,
                                                 RegressionNetEperiment,
                                                 SimpleNetExperiment)
from omsignal.experiments.lstm_experiment import LSTMExperiment
from omsignal.utils.loader import get_dataloader
from omsignal.utils.misc import check_file
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
        '--validation-data',
        type=str,
        help='Validation data file location',
        default=VALIDATION_LABELED_FILE,
    )

    parser.add_argument(
        '--train-data',
        type=str,
        help='Train data file location',
        default=TRAIN_LABELED_FILE,
    )

    parser.add_argument(
        '--params',
        type=str,
        help='Model param file location. '
        'For information about param file format refer README.md'
    )

    return parser


def train_cnn_classification(
        params,
        train_file=TRAIN_LABELED_FILE,
        validation_file=VALIDATION_LABELED_FILE):
    '''
    Main function
    '''
    model_file = MODEL_DIR / params['model_file']
    batch_size = params['batch_size']
    optimiser_params = params['optimiser_params']
    epochs = params['epochs']

    model_dir = os.path.dirname(os.path.realpath(model_file))
    os.makedirs(model_dir, exist_ok=True)

    # create id remap transformer
    remap = RemapLabels(ID_MAPPING)
    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        train_file,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        validation_file,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=SignalSegmenter(),
        shuffle=False,
        batch_size=batch_size
    )

    simplenet_exp = SimpleNetExperiment(
        model_file,
        optimiser_params=optimiser_params,
        device=device
    )
    print('started training')
    simplenet_exp.train(
        train_loader,
        epochs=epochs,
        validation_dataloader=valid_loader)
    remap.save()


def train_lstm_exp(
        params,
        train_file=TRAIN_LABELED_FILE,
        validation_file=VALIDATION_LABELED_FILE):
    """
    Main function
    """
    model_file = MODEL_DIR / params['model_file']
    batch_size = params['batch_size']
    optimiser_params = params['optimiser_params']
    model_params = params.get('model_params', {})
    epochs = params['epochs']

    model_dir = os.path.dirname(os.path.realpath(model_file))
    os.makedirs(model_dir, exist_ok=True)

    # remap labels
    remap = RemapLabels(ID_MAPPING)

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        train_file,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=LSTMSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        validation_file,
        torch.LongTensor,
        remap,
        only_ids=True,
        segmenter=LSTMSegmenter(),
        shuffle=False,
        batch_size=batch_size
    )

    model_params.update({
        'device': device
    })

    lstm_exp = LSTMExperiment(
        model_file,
        optimiser_params=optimiser_params,
        model_params=model_params,
        device=device
    )
    print('started training')
    lstm_exp.train(
        train_loader,
        epochs=epochs,
        validation_dataloader=valid_loader)
    remap.save()


def train_cnn_regression(
        params,
        train_file=TRAIN_LABELED_FILE,
        validation_file=VALIDATION_LABELED_FILE):
    '''
    Main function
    '''
    model_file = MODEL_DIR / params['model_file']
    batch_size = params['batch_size']
    optimiser_params = params['optimiser_params']
    epochs = params['epochs']

    model_dir = os.path.dirname(os.path.realpath(model_file))
    os.makedirs(model_dir, exist_ok=True)

    # remap labels
    remap = RemapLabels(ID_MAPPING)

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        train_file,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        validation_file,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )

    regnet_exp = RegressionNetEperiment(
        model_file,
        optimiser_params=optimiser_params,
        device=device
    )
    print('started training')
    regnet_exp.train(
        train_loader,
        epochs=epochs,
        validation_dataloader=valid_loader)
    remap.save()


def train_cnn_multi_task(
        params,
        train_file=TRAIN_LABELED_FILE,
        validation_file=VALIDATION_LABELED_FILE):
    '''
    Main function
    '''
    model_file = MODEL_DIR / params['model_file']
    batch_size = params['batch_size']
    optimiser_params = params['optimiser_params']
    epochs = params['epochs']

    model_dir = os.path.dirname(os.path.realpath(model_file))
    os.makedirs(model_dir, exist_ok=True)

    # remap labels
    remap = RemapLabels(ID_MAPPING)

    # create dataloaders
    train_loader, row_label_mapping_train = get_dataloader(
        train_file,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )
    valid_loader, row_label_mapping_valid = get_dataloader(
        validation_file,
        torch.FloatTensor,
        remap,
        only_ids=False,
        segmenter=SignalSegmenter(),
        shuffle=True,
        batch_size=batch_size
    )

    regnet_exp = MultiTaskExperiment(
        model_file,
        optimiser_params=optimiser_params,
        device=device
    )
    print('started training')
    regnet_exp.train(
        train_loader,
        epochs=epochs,
        validation_dataloader=valid_loader)
    remap.save()


# need to define below function definitions
MODEL_EXP_MAP = {
    'cnn_classification': {
        'train_func': train_cnn_classification,
        'param_file': PARAM_DIR / 'cnn_classification.yml',
    },
    'cnn_regression': {
        'train_func': train_cnn_regression,
        'param_file': PARAM_DIR / 'cnn_regression.yml',
    },
    'cnn_multi_task': {
        'train_func': train_cnn_multi_task,
        'param_file': PARAM_DIR / 'cnn_multi_task.yml',
    },
    'best_model': {
        'train_func': train_cnn_multi_task,
        'param_file': PARAM_DIR / 'cnn_multi_task.yml',
    },
}


def train(args):
    train_function = MODEL_EXP_MAP[args.model]['train_func']
    param_file = MODEL_EXP_MAP[args.model]['param_file']

    if args.params is not None:
        param_file = args.params
        param_file = check_file(param_file, PARAM_DIR)
    if param_file is None:
        exit(1)
    with open(param_file) as fob:
        params = yaml.load(fob)
    train_function(params, args.train_data, args.validation_data)


if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    train(args)

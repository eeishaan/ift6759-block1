#!/usr/bin/env python3
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from omsignal.constants import (MODEL_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import (MultiTaskExperiment,
                                                 RegressionNetEperiment,
                                                 SimpleNetExperiment)
from omsignal.experiments.lstm_experiment import LSTMExperiment
from omsignal.utils.loader import get_dataloader, get_vector_and_labels
from omsignal.utils.transform.basic import (LabelSeparator, RemapLabels,
                                            ToTensor)
from omsignal.utils.transform.preprocessor import (LSTMSegmenter, Preprocessor,
                                                   SignalSegmenter,
                                                   get_preprocessed_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_cnn_classification():
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


def run_lstm_exp():
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


def run_cnn_regression():
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


def run_cnn_multi_task():
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

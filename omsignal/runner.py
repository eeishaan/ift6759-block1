#!/usr/bin/env python3
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from omsignal.constants import (MODEL_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import SimpleNetExperiment
from omsignal.utils.loader import read_memfile
from omsignal.utils.transform.basic import (LabelSeparator, RemapLabels,
                                            ToTensor)
from omsignal.utils.transform.preprocessor import Preprocessor
from omsignal.utils.transform.signal import SignalSegmenter
from omsignal.utils.loader import OmsignalDataset, get_dataloader


def run_cnn_exp():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "simple_net.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    remap_transform = RemapLabels()
    segment_size = 110
    transform = Compose([
        remap_transform,
        ToTensor(),
        LabelSeparator(),
        Preprocessor(),
        SignalSegmenter(segment_size),
    ])

    # initialize train dataloader
    train_dataset = OmsignalDataset(
        TRAIN_LABELED_FILE,
        transform=transform)
    train_dataloader = get_dataloader(
        train_dataset, batch_size=3, num_workers=4, shuffle=True)

    # initialize validation dataloader
    validation_dataset = OmsignalDataset(
        VALIDATION_LABELED_FILE,
        transform=transform)
    validation_dataloader = get_dataloader(
        validation_dataset, batch_size=3, num_workers=4, shuffle=True)

    simplenet_exp = SimpleNetExperiment(
        model_file,
        optimiser_params={
            'lr': 0.1
        },
        device=device
    )
    print('started training')
    simplenet_exp.train(
        train_dataloader,
        epochs=3000,
        validation_dataloader=validation_dataloader)

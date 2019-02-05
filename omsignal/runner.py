#!/usr/bin/env python3
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from omsignal.constants import (MODEL_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import SimpleNetExperiment
from omsignal.utils.loader import (OmsignalDataset, get_dataloader,
                                   get_vector_and_labels, read_memfile)
from omsignal.utils.transform.basic import (LabelSeparator, RemapLabels,
                                            ToTensor)
from omsignal.utils.transform.preprocessor import Preprocessor, SignalSegmenter


def run_cnn_exp():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "simple_net.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_data, train_labels = get_vector_and_labels(TRAIN_LABELED_FILE)
    valid_data, valid_labels = get_vector_and_labels(VALIDATION_LABELED_FILE)

    # run preprocessing
    preprocessor = Preprocessor()
    train_data = torch.tensor(train_data).to(device)
    valid_data = torch.tensor(valid_data).to(device)
    train_data = preprocessor(train_data)
    valid_data = preprocessor(valid_data)

    # remap labels
    remap = RemapLabels()
    train_labels = np.apply_along_axis(remap, 1, train_labels)
    valid_labels = np.apply_along_axis(remap, 1, valid_labels)

    # create segments
    segmenter = SignalSegmenter()
    train_data, train_ids = segmenter(train_data)
    valid_data, valid_ids = segmenter(valid_data)

    # create a second level of label mapping
    row_label_mapping_train = {i: j for i, j in enumerate(train_labels[:, -1])}
    row_label_mapping_valid = {i: j for i, j in enumerate(valid_labels[:, -1])}

    train_labels = np.array([row_label_mapping_train[i] for i in train_ids])
    valid_labels = np.array([row_label_mapping_valid[i] for i in valid_ids])

    # create dataloaders
    train_data = torch.Tensor(train_data)
    train_labels = torch.LongTensor(train_labels)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)

    valid_data = torch.Tensor(valid_data)
    valid_labels = torch.LongTensor(valid_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=128, shuffle=False)

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

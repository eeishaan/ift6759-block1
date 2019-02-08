#!/usr/bin/env python3
'''
Data loading routines
'''

import torch
from torch.utils.data import DataLoader, Dataset

from omsignal.utils.memfile import read_memfile
from omsignal.utils.transform.preprocessor import get_preprocessed_data
from omsignal.utils.misc import check_file
from omsignal.constants import DATA_ROOT_DIR


class OmsignalDataset(Dataset):
    '''
    Omsignal dataset class
    '''

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        label = self.labels[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def get_dataloader(
        data_file_path,
        label_type,
        remap_transform,
        only_ids,
        segmenter,
        shuffle,
        batch_size):
    '''
    Construct dataloader for train/validation data
    '''
    # load data
    data, labels = get_vector_and_labels(data_file_path)

    # get_preprocess_data
    data, labels, row_id_map = get_preprocessed_data(
        data,
        labels,
        only_ids,
        remap_transform,
        segmenter
    )

    # make dataloader
    data = torch.Tensor(data)
    labels = label_type(labels)
    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, row_id_map


def get_test_dataloader(test_data_file, segmenter, batch_size=160):
    '''
    Construct dataloader for test data
    '''

    # get correct file path
    test_data_file = check_file(test_data_file, DATA_ROOT_DIR)
    if test_data_file is None:
        return

    # load test data
    test_data = read_memfile(
        test_data_file, shape=(160, 3750), dtype='float32')

    # pre-process
    test_data, row_mapping = get_preprocessed_data(
        data=test_data,
        labels=None,
        only_ids=False,
        remap_label_transformer=None,
        segmenter=segmenter,
    )

    # make dataloader
    test_data = torch.Tensor(test_data)
    dataset = torch.utils.data.TensorDataset(test_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    return loader, row_mapping


def get_vector_and_labels(file_path: str) -> tuple:
    """Separate raw data and labels from the specified memfile path

    Args:
        file_path: numpy memfile file containing the data

    Returns:
        tuple: (raw_data_ndarray, label_ndarray)

        First element contains the raw data ndarray, second element
        is the label ndarray
    """
    dataset = read_memfile(file_path, shape=(160, 3754), dtype='float32')
    return dataset[:, :-4], dataset[:, -4:]

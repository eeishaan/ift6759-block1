#!/usr/bin/env python3
'''
Data loading routines
'''

from torch.utils.data import DataLoader, Dataset

from omsignal.utils.memfile import read_memfile
from omsignal.utils.transform.preprocessor import get_preprocessed_data


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
        segmenter_cls,
        shuffle,
        batch_size):
    # load data
    train_data, train_labels = get_vector_and_labels(data_file_path)

    # get_preprocess_data
    train_data, train_labels, row_id_map = get_preprocessed_data(
        train_data,
        train_labels,
        only_ids,
        remap_transform,
        segmenter_cls
    )

    # make dataloader
    train_data = torch.Tensor(train_data)
    train_labels = label_type(train_labels)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, row_id_map


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

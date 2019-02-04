#!/usr/bin/env python3
'''
Data loading routines
'''

from torch.utils.data import DataLoader, Dataset

from omsignal.utils.memfile import read_memfile


class OmsignalDataset(Dataset):
    '''
    Omsignal dataset class
    '''

    def __init__(self, file_path,
            shape=(160, 3754), dtype='float32', transform=None):
        self.data = read_memfile(file_path, shape=shape, dtype=dtype)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_dataloader(dataset, shuffle=True, batch_size=20, num_workers=4):
    '''
    Construct a dataloader
    '''
    return DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)


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

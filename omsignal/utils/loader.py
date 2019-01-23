#!/usr/bin/env python3
'''
Data loading routines
'''

from omsignal.utils.memfile import read_memfile


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

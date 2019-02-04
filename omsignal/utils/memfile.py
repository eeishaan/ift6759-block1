#!/usr/bin/env python3
'''
Module for reading/writing numpy memfile
'''

import numpy as np


def read_memfile(filename: str, shape: tuple, dtype: str = 'float32') -> np.ndarray:
    """Read binary data and return as a numpy array of desired shape

    Args:
        filename: Path of memfile.
        shape: Shape of numpy array.
        dtype (:obj:`str`, optional): numpy dtype. Defaults to ``float32``.

    Returns:
        ndarray: A numpy ndarray with data from memfile.
    """
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    data = np.zeros(shape=shape, dtype=dtype)
    data[:] = fp[:]
    del fp
    return data


def write_memfile(data: np.ndarray, filename: str) -> None:
    """Writes ``data`` to file specified by ``filename``.

    Args:
        data: ndarray containing the data.
        filename: Name of memfile to be created with contents of ``data``.

    Returns:
        None
    """
    shape = data.shape
    dtype = data.dtype
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    fp[:] = data[:]
    del fp

#!/usr/bin/env python3
'''
Module containing data transformation
'''

import numpy as np
import torch
from scipy import signal


class ToTensor(object):
    '''
    Transformer for converting numpy array to a tensor object
    '''

    def __init__(self):
        pass
    
    def __call__(self, X):
        return torch.from_numpy(X)


class ToNumpy(object):
    '''
    Transformer for converting tensor object to a numpy object
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return X.numpy()

class RemapLabels(object):
    def __init__(self):
        self.map = {}
        self.idx = -1

    def __call__(self, X):
        label = X[-1]        
        if label not in self.map:
            self.idx +=1
            self.map[label] = self.idx
        X[-1] = self.map[label]
        return X

class FFT(object):
    '''
    Fast fourier trasnform
    '''

    def __init__(self, signal_ndim=1, normalized=False):
        self.signal_ndim = signal_ndim
        self.normalized = normalized

    def __call__(self, X):
        # add a new dimesion and fill it with zero
        X.unsqueeze_(2)
        X.index_fill_(1, torch.tensor([2]), 0)
        out = torch.fft(X, 1)
        return out
    

class Spectogram(object):
    '''
    Spectogram transform
    '''

    def __init__(self, lognorm: bool=False, fs: int=16,
                 nperseg: int=256, noverlap: int=None):
        '''Takes a time-series x and returns the spectogram
            Args:
                lognorm: log spectrogram or not. Defaults to ``False``
                fs: Sampling frequency of the x time series.
                    Defaults to 16.
                nperseg: Length of each segment. Defaults to 256.
                noverlap: Number of points to overlap between segments.
                        If None, ``noverlap = nperseg // 8``. Defaults to None.

            Returns :
                  f (ndarray): Array of sample frequencies.
                  t (ndarray): Array of segment times.
                  Zxx (ndarray): Spectrogram of x.
                    By default, the last axis of Zxx corresponds
                    to the segment times.
        '''
        self.lognorm = lognorm
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __call__(self, X):
        f, t, Zxx = signal.spectrogram(
            X, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap
        )

        if self.lognorm:
            Zxx = np.abs(Zxx)
            mask = Zxx > 0
            Zxx[mask] = np.log(Zxx[mask])
            Zxx = (Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx))

        return f, t, Zxx

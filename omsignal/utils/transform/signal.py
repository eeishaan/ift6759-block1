#!/usr/bin/env python3
'''
Transform signal
'''

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from omsignal.utils.transform import OmTransform


class Invert(OmTransform):
    '''
    Invert the signal
    '''

    def __init__(self):
        pass

    def __call__(self, X):
        return X * -1


class FFT(OmTransform):
    '''
    Fast fourier trasnform
    '''

    def __init__(self, signal_ndim=1, normalized=True):
        self.signal_ndim = signal_ndim
        self.normalized = normalized

    def __call__(self, X):
        # takes a time-series x and returns the FFT
        # input: x
        # output : R and I; real and imaginary componenet of the real FFT
        y = np.fft.rfft(X, norm='ortho')

        real, im = np.real(y), np.imag(y)

        if self.normalized:
            real = real/np.max(real)
            im = im/np.max(im)

        fourier = real**2 + im**2
        fourier = torch.from_numpy(fourier)
        return fourier


class SignalShift(OmTransform):
    '''
    Shift the signal 
    '''

    def __init__(self, data_len=-4, shift_len=10, dim=-1):
        self.data_len = data_len
        self.shift_len = shift_len
        self.dim = dim

    def __call__(self, X):
        num_shits = int(X.shape[-1]/self.shift_len)

        def _augment_and_duplicate_labels(sample):
            if len(sample.shape) == 2:
                raw_data, labels = sample[:,
                                          :self.data_len], sample[:, self.data_len:]
            else:
                raw_data, labels = sample[:self.data_len], sample[self.data_len:]
            new_data = np.roll(raw_data, shift=self.shift_len, axis=self.dim)
            return np.hstack((new_data, labels))
        new_data = np.vstack([_augment_and_duplicate_labels(X)
                              for _ in range(num_shits)])
        return new_data


class Spectogram(OmTransform):
    '''
    Spectogram transform
    '''

    def __init__(self, lognorm: bool = False, fs: int = 16,
                 nperseg: int = 256, noverlap: int = None):
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

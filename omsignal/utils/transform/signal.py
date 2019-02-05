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

    def __init__(self, signal_ndim=1, normalized=False):
        self.signal_ndim = signal_ndim
        self.normalized = normalized

    def __call__(self, X):
        # add a new dimesion and fill it with zero
        X.unsqueeze_(2)
        X.index_fill_(1, torch.tensor([2]), 0)
        out = torch.fft(X, 1)
        return out


class SignalSegmenter(OmTransform):

    def __init__(self, segment_size=110):
        self.segment_size = segment_size

    @classmethod
    def detect_R_peak(cls, data, sada_wd_size=7, fs_wd_size=12, threshold=35):
        """
        Take a Batch of ECG data and find the location of the R Peak

        The algorithm is based on the paper:
        Online and Offline Determination of QT and PR Interval and QRS Duration in Electrocardiography
        (Bachler et al., 2012)
        The variable name and default value follow the paper

        Parameters
        ----------
        data : numpy array
            The ECG Data (batch size x lenght of the ECG recording)
        sada_wd_size: int
            size of the moving window used in the calculation of SA and DA
        fs_wd_size: int
            size of the moving window used in the calculation of the feature signal FS
        threshold: int
            FS is compared to the threshold to determined if its a QRS zone. 
        """
        R_peak = []

        # Allow batch size of 1
        if len(data.size()) == 1:
            data = data.unsqueeze(0)

        D = data[:, 1:] - data[:, 0:-1]

        data = data.unsqueeze(0)
        D = D.unsqueeze(0)
        SA = F.max_pool1d(data, kernel_size=sada_wd_size, stride=1)
        SA = SA + F.max_pool1d(-data, kernel_size=sada_wd_size, stride=1)
        DA = F.max_pool1d(D, kernel_size=sada_wd_size, stride=1, padding=1)
        DA = DA + F.max_pool1d(-D, kernel_size=sada_wd_size,
                               stride=1, padding=1)

        C = DA[:, :, 1:] * torch.pow(SA, 2)
        FS = F.max_pool1d(C, kernel_size=fs_wd_size, stride=1)
        detect_filter = (FS > threshold)

        detect_filter = detect_filter.squeeze(0).cpu()
        data = data.squeeze(0).cpu()

        for ECG in range(len(data)):

            in_QRS = 0
            start_QRS = 0
            end_QRS = 0
            r_peak = np.array([])

            for tick, detect in enumerate(detect_filter[ECG]):

                if (in_QRS == 0) and (detect == 1):
                    start_QRS = tick
                    in_QRS = 1

                elif (in_QRS == 1) and (detect == 0):
                    end_QRS = tick
                    R_tick = torch.argmax(
                        data[ECG, start_QRS: end_QRS+sada_wd_size+fs_wd_size]).item()
                    r_peak = np.append(r_peak, R_tick + start_QRS)
                    in_QRS = 0
                    start_QRS = 0

            R_peak.append(r_peak)

        return R_peak

    def __call__(self, *args):
        data, label = args[0]
        data = data.unsqueeze(0)
        label = label.unsqueeze(0)

        R_peak = SignalSegmenter.detect_R_peak(data)

        template_found = 0
        max_len = 200
        all_templates = data.new_empty((max_len, self.segment_size))
        for recording in range(len(R_peak)):
            # generate the template
            half_size_int = int(self.segment_size//2)

            for i in R_peak[recording][1:-1]:
                new_heart_beat = data[recording][int(
                    i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                if len(new_heart_beat) == 0:
                    continue
                if len(new_heart_beat) != self.segment_size:
                    # TODO: pad the seq
                    continue
                all_templates[template_found, :] = new_heart_beat
                template_found += 1
                if template_found == max_len:
                    break
            if template_found == max_len:
                break

        all_labels = label.repeat(template_found, 1)
        if template_found != max_len:
            all_labels = torch.cat((
                all_labels,
                label.new_full((max_len-template_found, label.shape[-1]), -1)))
        return all_templates, all_labels


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
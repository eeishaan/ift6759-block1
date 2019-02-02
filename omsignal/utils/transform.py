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
        X = np.copy(X)
        label = str(int(X[-1]))
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


class SignalSegmenter(object):

    def __init__(self, R_peak, segment_size=110, all_window=True):
        self.segment_size = segment_size
        self.all_window = all_window

    @classmethod
    def detect_R_peak(data, SADA_wd_size = 7, FS_wd_size = 12, Threshold = 35):
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
        SADA_wd_size: int
            size of the moving window used in the calculation of SA and DA
        FS_wd_size: int
            size of the moving window used in the calculation of the feature signal FS
        Threshold: int
            FS is compared to the Threshold to determined if its a QRS zone. 
        """
        R_peak = []
        
        #Allow batch size of 1
        if len(data.size()) == 1:
            data = data.unsqueeze(0)
        
        D = data[:, 1:] - data[:, 0:-1]
        
        
        data = data.unsqueeze(0)
        D = D.unsqueeze(0)
        SA = F.max_pool1d(data, kernel_size = SADA_wd_size, stride = 1)
        SA = SA + F.max_pool1d(-data, kernel_size = SADA_wd_size, stride = 1) 
        DA = F.max_pool1d(D, kernel_size = SADA_wd_size, stride = 1, padding=1)
        DA = DA + F.max_pool1d(-D, kernel_size = SADA_wd_size, stride = 1, padding=1) 
        
        C = DA[:,:,1:] * torch.pow(SA, 2)
        FS = F.max_pool1d(C, kernel_size = FS_wd_size, stride = 1) 
        Detect = (FS > Threshold)
        
        Detect = Detect.squeeze(0).cpu()
        data = data.squeeze(0).cpu()

        for ECG in range(len(data)):
            
            in_QRS = 0
            start_QRS = 0
            end_QRS = 0
            r_peak = np.array([])
            
            for tick, detect in enumerate(Detect[ECG]):
                
                if (in_QRS == 0) and (detect == 1):
                    start_QRS = tick
                    in_QRS = 1
                    
                elif (in_QRS == 1) and (detect == 0):
                    end_QRS = tick
                    R_tick = torch.argmax(data[ECG, start_QRS : end_QRS+SADA_wd_size+FS_wd_size]).item()
                    r_peak = np.append(r_peak, R_tick + start_QRS)
                    in_QRS = 0
                    start_QRS = 0
                    
            R_peak.append(r_peak)
            
        return R_peak

    def __call__(self, data):
        R_peak = SignalSegmenter.detect_R_peak(data)

        listoftemplate = np.empty((1,self.segment_size))
        
        for recording in range(len(R_peak)):
            #generate the template
            half_size_int = int(self.segment_size//2)
            template = np.zeros((1,self.segment_size))
            
            for i in R_peak[recording][1:-1]:
                new_heart_beat = data[recording][int(i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                #add padding to make them all of the same size
                current_size = int(half_size_int*0.8)+int(half_size_int*1.2)
                
                    
                template = np.concatenate((template,
                                        np.expand_dims(new_heart_beat, axis = 0))
                                        , axis=0)

            if self.all_window == False:
                template = np.mean(template, axis = 0)
                template = np.expand_dims(template, axis = 0)
            
            listoftemplate = np.append(listoftemplate, template, axis = 0)
        
        return listoftemplate[1:]

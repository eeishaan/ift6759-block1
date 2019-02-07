#!/usr/bin/evn python3

import numpy as np
import torch
import torch.nn.functional as F


def detect_R_peak(data, sada_wd_size=7, fs_wd_size=12, threshold=35):
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
    r_peaks = []

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

        r_peaks.append(r_peak)

    return r_peaks


def rr_mean_std(r_peak, max_interval=180):
    """
    Calculate the mean RR interval and the std

    Parameters
    ----------
    R_peak : list of list
        Each entry is a list of the positiion of the R peak in the ECG
    MaxInterval: int
        maximum lenght of an interval, interval higher than this amount are ignore
    """
    # calculate the lenght of the interval
    rr_interval = [r_peak[i][1:]-r_peak[i][0:-1] for i in range(len(r_peak))]

    # We keep only good quality one
    rr_interval_adj = [interval[interval < 180] for interval in rr_interval]

    rr_interval_mean = [np.mean(interval) for interval in rr_interval_adj]
    rr_std = [np.std(interval) for interval in rr_interval_adj]

    return rr_interval_mean, rr_std

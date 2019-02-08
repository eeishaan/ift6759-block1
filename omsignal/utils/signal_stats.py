#!/usr/bin/evn python3

import numpy as np
import torch
import torch.nn.functional as F


def detect_R_peak(data, sada_wd_size=7, fs_wd_size=12, threshold=35):
    """
    Take a Batch of ECG data and find the location of the R peak

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


def fing_peak_offset_onset(heart_beat, side, half_size_int, c1, c2):
    """
    Take an heart_beat an find the points of interest at Left or Right of the R peak

    The algorithm is based on the paper:
    Online and Offline Determination of QT and PR Interval and QRS Duration in Electrocardiography
    (Bachler et al., 2012)
    The variable name and default value follow the paper. 

    Parameters
    ----------
    heart_beat : list
        Contains the information of one hearth beat
    side: Left or Right
        Left to detect T peak T offset QRS offset, Right to detect P onset, P peak, QRS onset
    half_size_int: int
        half of the RR Mean interval
    c1 : Float
        Used to determined the threshold for QRS offset, onset detection (change in the signal)
    c2 : Float
        Used to determined the threshold for QRS offset, onset detection (change in the derivative of the signal)

    Output
    ----------
    If side = Left Position of the QRS offset, T peak, T offset 
    If side = Right Position of the  QRS onset, P peak, P onset  
    """

    hb_length = len(heart_beat)

    if side == "Right":
        # reverse the template to detect points related to the P wave
        heart_beat = np.flip(heart_beat)

    # find QRS offset/QRS Onset
    derivative = heart_beat[1:] - heart_beat[:-1]
    TA = np.array([])
    TDA = np.array([])

    for i in range(int(half_size_int * 0.8) + 1, int(half_size_int * 0.8) + 25):
        TA = np.append(TA, np.max(
            heart_beat[i: i+4]) - np.min(heart_beat[i: i+4]))
        TDA = np.append(TDA, np.max(
            derivative[i: i+4]) - np.min(derivative[i: i+4]))

    TT = c1 * (np.max(TA) - np.min(TA)) + np.min(TA)
    TD = c2 * (np.max(TDA) - np.min(TDA)) + np.min(TDA)

    for i in range(0, len(TA)):
        if TA[i] < TT or TDA[i] < TD:

            if side == "Right":
                QRS_offset = i + int(half_size_int*1.2) + 4
                break

            else:
                QRS_offset = i + int(half_size_int*0.8) + 4
                break

    # find T peak/ P peak
    peak = QRS_offset + np.argmax(heart_beat[QRS_offset:])

    # find T wave offset/P wave onset see paper for details on how it works

    k = (heart_beat[peak] - heart_beat[-1]) / (peak-hb_length)
    d = heart_beat[-1] - k*hb_length
    g = k*np.arange(0, hb_length) + d
    decision = heart_beat - g
    peak_offset = np.argmin(decision[peak:hb_length]) + peak

    if side == "Right":
        # reverse the points to get the points in the original hearth beat
        QRS_offset = hb_length - QRS_offset - 1
        peak = hb_length - peak - 1
        peak_offset = hb_length - peak_offset - 1

    return QRS_offset, peak, peak_offset


def find_ecg_points(data, R_peak, RR_Mean_Interval):
    """
    Take a Batch of ECG data and find the location of all points

    The algorithm is based on the paper:
    Online and Offline Determination of QT and PR Interval and QRS Duration in Electrocardiography
    (Bachler et al., 2012)
    The variable name and default value follow the paper

    Parameters
    ----------
    data : numpy array
        The ECG Data (batch size x lenght of the ECG recording)
    R_peak: list of list
        List of batch size lenght that contain the contain list of the location of the R peak
    RR_Mean_Interval: list
        List of batch size lenght that contains the Mean RR interval of each ECG

    Output
    ----------
    Dictionnary with the location of all points
    """

    ECG_points = {}
    ECG_points["Hearth_Beat"] = []
    ECG_points["R_Peak"] = []
    ECG_points["P_Peak"] = []
    ECG_points["T_Peak"] = []
    ECG_points["QRS_offset"] = []
    ECG_points["QRS_onset"] = []
    ECG_points["P_onset"] = []
    ECG_points["T_offset"] = []

    for recording in range(len(data)):
        # generate the averaged hearth beat
        # Each hearth beat is of variable size
        half_size_int = int(RR_Mean_Interval[recording]//2)
        segments = []

        for i in R_peak[recording][1:-1]:
            new_heart_beat = data[recording][int(
                i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
            segments.append(new_heart_beat)

        avg_segments = np.mean(segments, axis=0)

        QRS_offset, T_peak, T_offset = fing_peak_offset_onset(
            avg_segments, "Left", half_size_int, 0.1, 0.1)

        QRS_onset, P_peak, P_onset = fing_peak_offset_onset(
            avg_segments, "Right", half_size_int, 0.5, 0.5)

        ECG_points["Hearth_Beat"].append(avg_segments)
        ECG_points["R_Peak"].append(int(half_size_int*0.8))
        ECG_points["P_Peak"].append(P_peak)
        ECG_points["T_Peak"].append(T_peak)
        ECG_points["QRS_offset"].append(QRS_offset)
        ECG_points["QRS_onset"].append(QRS_onset)
        ECG_points["P_onset"].append(P_onset)
        ECG_points["T_offset"].append(T_offset)

    return ECG_points


def rt_mean_pr_mean(ECG_points):
    """
    Calculate the mean RT Mean interval and PR_Mean

    Parameters
    ----------
    ECG_points : Dictionnary
        Contains the location of all point and the hearth beat data
    """
    RT_Mean = [T_peak-R_peak for T_peak,
               R_peak in zip(ECG_points["T_Peak"], ECG_points["R_Peak"])]
    PR_Mean = [R_peak-P_peak for R_peak,
               P_peak in zip(ECG_points["R_Peak"], ECG_points["P_Peak"])]

    return RT_Mean, PR_Mean


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

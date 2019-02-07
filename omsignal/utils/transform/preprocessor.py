import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omsignal.utils.transform.signal import FFT


class Preprocessor(object):

    def __init__(
            self,
            ma_window_size=2,
            mv_window_size=4,
            num_samples_per_second=125):
        # ma_window_size: (in seconds) window size to use
        #                 for moving average baseline wander removal
        # mv_window_size: (in seconds) window size to use
        #                 for moving average RMS normalization

        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd

        self.maKernelSize = (ma_window_size * num_samples_per_second) + 1

        # Kernel size to use for moving average normalization: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd

        self.mvKernelSize = (mv_window_size * num_samples_per_second) + 1

    def __call__(self, *args):

        if len(args[0]) == 2:
            x, label = args[0]
            x = x.view(1, 1, -1)
        else:
            x = args[0]
            x = x.unsqueeze(0)

        # Remove window mean and standard deviation

        x = (x - torch.mean(x, dim=2, keepdim=True)) / \
            (torch.std(x, dim=2, keepdim=True) + 0.00001)

        # Moving average baseline wander removal

        x = x - F.avg_pool1d(
            x, kernel_size=self.maKernelSize,
            stride=1, padding=(self.maKernelSize - 1) // 2
        )

        # Moving RMS normalization

        x = x / (
            torch.sqrt(
                F.avg_pool1d(
                    torch.pow(x, 2),
                    kernel_size=self.mvKernelSize,
                    stride=1, padding=(self.mvKernelSize - 1) // 2
                )) + 0.00001
        )

        # Don't backpropagate further

        if len(args[0]) == 2:
            x = x.view(-1)
            return x, label
        else:
            return x.squeeze(0)


class SignalSegmenter(object):

    def __init__(self, segment_size=110, take_average=False):
        self.segment_size = segment_size
        self.take_average = take_average

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

    def __call__(self, data):

        r_peak = SignalSegmenter.detect_R_peak(data)
        data = data.cpu().numpy()
        all_segments = []
        all_ecg_ids = []
        half_size_int = int(self.segment_size//2)

        for recording in range(len(data)):
            segments = []
            ecg_ids = []
            for i in r_peak[recording][1: -1]:
                new_heart_beat = data[recording][int(
                    i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                if len(new_heart_beat) != 110:
                    continue
                segments.append(new_heart_beat)
                ecg_ids.append(recording)

            if self.take_average is True:
                segments = [np.mean(segments, axis=0)]
                ecg_ids = [int(np.mean(ecg_ids))]
            all_segments.extend(segments)
            all_ecg_ids.extend(ecg_ids)

        return np.array(all_segments), np.array(all_ecg_ids)

class LSTM_Segmenter():
    def __init__(self, top_freq=3, max_hb=15, segment_size=110):
        self.segment_size = segment_size

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

    def Get_RR_Mean_std(R_peak, MaxInterval=180):
    """
    Calculate the mean RR interval and the std
    
    Parameters
    ----------
    R_peak : list of list
        Each entry is a list of the positiion of the R peak in the ECG
    MaxInterval: int
        maximum lenght of an interval, interval higher than this amount are ignore
    """
    #calculate the lenght of the interval
    RR_interval = [R_peak[i][1:]-R_peak[i][0:-1] for i in range(len(R_peak))]
    
    #We keep only good quality one
    RR_interval_adj = [interval[interval<180] for interval in RR_interval]
    

    RR_interval_mean = [np.mean(interval) for interval in RR_interval_adj]    
    RR_std = [np.std(interval) for interval in RR_interval_adj]
    
    return RR_interval_mean, RR_std

    def __call__(self, data):
        R_peak = self.detect_R_peak(data)
        fft = FFT()
        heartb = []
        half_size_int = self.segment_size//2
        ECG_id = []
        ecg_idx = 0

        RR_mean, RR_std = self.Get_RR_Mean_std(R_peak)
        
        for ecg in data:
            fourier = fft(ecg)
            _, high_frequency = torch.topk(fourier, self.top_freq)
            rr_mean = torch.tensor([RR_mean[ecg_idx]])
            rr_std = torch.tensor([RR_std[ecg_idx]])
            features = torch.cat((high_frequency.float(), rr_mean, rr_std))

            nfeat = len(features)

            heartbeats = torch.zeros((1, self.max_hb, self.segment_size+nfeat))
            hb_idx = 0
            #generate the template             
            for i in R_peak[ecg_idx][1:-1]:
                if hb_idx >= self.max_hb:
                    break
                heartbeat = ecg[int(
                     i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                heartbeats[0, hb_idx, :self.segment_size] = heartbeat
                heartbeats[0, hb_idx, self.segment_size:] = features
                
                hb_idx += 1              

            #template[:][:-top_freq] -= np.mean(template[:][:-top_freq], axis=0)
            #template[:][:-top_freq] /= np.std(template[:][:-top_freq], axis=0)+0.000001
            heartb.append(heartbeats)
            ECG_id.append(ecg_idx) 
            ecg_idx += 1

        heartb = tuple(heartb)
        heartb = torch.cat(heartb)

        heartb[:, :, self.segment_size:] -= torch.mean(heartb[:, :, self.segment_size:],
            dim=0)
        heartb[:, :, self.segment_size:] /= torch.std(heartb[:, :, self.segment_size:],
            dim=0)+0.000001

        ECG_id = torch.tensor(ECG_id)
            
        #listofheartb = [hearth_beat for hearth_beat in template for template in listofheartb]
        return heartb, ECG_id

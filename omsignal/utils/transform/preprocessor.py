import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __call__(self, data):

        r_peak = SignalSegmenter.detect_R_peak(data)

        all_segments = []
        all_ecg_ids = []
        half_size_int = int(self.segment_size//2)

        for recording in range(len(data)):
            segments = []
            ecg_ids = []
            for i in r_peak[recording][1:-1]:
                new_heart_beat = data[recording][int(
                    i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                if len(new_heart_beat) != 110:
                    continue
                segments.append(new_heart_beat)
                ecg_ids.append(recording)

            all_segments.append(segments)
            all_ecg_ids.append(ecg_ids)

        #listoftemplate = [hearth_beat for hearth_beat in template for template in listoftemplate]
        return np.array(all_segments), np.array(all_ecg_ids)

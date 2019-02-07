import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omsignal.utils.transform.signal import FFT
from omsignal.utils.signal_stats import detect_R_peak, rr_mean_std


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

    def __call__(self, data):
        r_peak = detect_R_peak(data)
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


class LSTMSegmenter():
    def __init__(self, top_freq=3, max_hb=15, segment_size=110):
        self.segment_size = segment_size
        self.top_freq = top_freq
        self.max_hb = max_hb

    def __call__(self, data):
        r_peak = detect_R_peak(data)
        fft = FFT()
        heartb = []
        half_size_int = self.segment_size//2
        ecg_ids = []
        ecg_idx = 0

        rr_mean_list, rr_std_list = rr_mean_std(r_peak)

        for ecg in data:
            fourier = fft(ecg)
            _, high_frequency = torch.topk(fourier, self.top_freq)
            rr_mean = torch.tensor([rr_mean_list[ecg_idx]])
            rr_std = torch.tensor([rr_std_list[ecg_idx]])
            features = torch.cat((high_frequency.float(), rr_mean, rr_std))

            nfeat = len(features)

            heartbeats = torch.zeros((1, self.max_hb, self.segment_size+nfeat))
            hb_idx = 0
            # generate the template
            for i in r_peak[ecg_idx][1:-1]:
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
            ecg_ids.append(ecg_idx)
            ecg_idx += 1

        heartb = tuple(heartb)
        heartb = torch.cat(heartb)

        heartb[:, :, self.segment_size:] -= torch.mean(heartb[:, :, self.segment_size:],
                                                       dim=0)
        heartb[:, :, self.segment_size:] /= torch.std(heartb[:, :, self.segment_size:],
                                                      dim=0)+0.000001

        ecg_ids = torch.tensor(ecg_ids)

        #listofheartb = [hearth_beat for hearth_beat in template for template in listofheartb]
        return heartb, ecg_ids


def get_preprocessed_data(
        data,
        labels,
        only_ids,
        remap_label_transformer,
        segmenter):

    # run preprocessing
    preprocessor = Preprocessor()
    data = torch.tensor(data)
    data = preprocessor(data)

    # remap labels
    if labels:
        labels = np.apply_along_axis(
            remap_label_transformer, 1, labels)

    # create segments
    data, train_ids = segmenter(data)

    # create a second level of label mapping
    if labels:
        row_label_mapping_train = {i: j for i, j in enumerate(labels[:, -1])}
        if only_ids is True:
            labels = np.array([row_label_mapping_train[i]
                               for i in train_ids])
        else:
            labels = np.array([
                np.hstack((labels[i][:-1], [row_label_mapping_train[i]]))
                for i in train_ids
            ])
        return data, labels, row_label_mapping_train
    return data, train_ids

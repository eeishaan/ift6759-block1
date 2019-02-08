#!/usr/bin/evn python3
import logging
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from omsignal.utils.score import get_combined_score
from omsignal.utils.signal_stats import (detect_R_peak, find_ecg_points,
                                         rr_mean_std, rt_mean_pr_mean)
from omsignal.utils.transform.preprocessor import SignalSegmenter

logger = logging.getLogger(__name__)


class DeterministicExp(object):
    def __init__(self, exp_file):
        self._exp_file = exp_file
        self.ipca = PCA(n_components=30)
        self.lda = LinearDiscriminantAnalysis()

    def eval(self, valid_data, valid_labels):
        predicted_labels = self.test(valid_data)
        message = "Eval Stats \n"
        message += get_combined_score(
            predicted_labels[:, 0],
            predicted_labels[:, 1],
            predicted_labels[:, 2],
            predicted_labels[:, 3],
            valid_labels)
        logger.info(message)

    @classmethod
    def load_experiment(cls, file_name):
        obj = cls(file_name)
        with open(file_name, 'rb') as fob:
            save_dict = pickle.load(fob)
        obj.ipca = save_dict['ipca']
        obj.lda = save_dict['lda']
        return obj

    def test(self, data):
        # detect the R peak
        r_peak = detect_R_peak(data)

        # get the RR mean and std-dev
        rr_mean, rr_std = rr_mean_std(r_peak)

        # find the rt_mean and pr_mean
        ecg_points = find_ecg_points(
            data, r_peak, rr_mean)

        rt_mean, pr_mean = rt_mean_pr_mean(ecg_points)

        # take average segments
        segmenter = SignalSegmenter(take_average=True)
        data, _ = segmenter(data)

        data_pca = self.ipca.transform(data)
        id_preds = self.lda.predict(data_pca)

        return np.hstack((
            np.array(pr_mean).reshape(-1, 1),
            np.array(rt_mean).reshape(-1, 1),
            np.array(rr_std).reshape(-1, 1),
            np.array(id_preds).reshape(-1, 1),
        ))

    def train(self, train_data, train_labels, valid_data=None, valid_labels=None):
        train_data_orig = train_data
        train_labels_orig = train_labels

        # take average segments
        segmenter = SignalSegmenter(take_average=True)
        train_data, _ = segmenter(train_data)
        valid_data, _ = segmenter(valid_data)

        train_ids = train_labels[:, -1]

        shuffle = np.random.permutation(len(train_data))
        train_data = train_data[shuffle]
        train_ids = train_ids[shuffle]

        self.ipca.fit(train_data)

        pca_train = self.ipca.transform(train_data)

        self.lda.fit(pca_train, train_ids)

        predicted_train_labels = self.test(train_data_orig)
        message = "Train Stats \n"
        message += get_combined_score(
            predicted_train_labels[:, 0],
            predicted_train_labels[:, 1],
            predicted_train_labels[:, 2],
            predicted_train_labels[:, 3],
            train_labels_orig)
        logger.info(message)
        self.eval(valid_data, valid_labels)

    def save_experiment(self):
        save_dict = {
            'ipca': self.ipca,
            'lda': self.lda
        }
        with open(self._exp_file) as fob:
            pickle.dump(save_dict, fob)

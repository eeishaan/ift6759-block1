#!/usr/bin/env python3

from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, recall_score

import numpy as np


def get_combined_score(pr_mean, tr_mean, rr_std, pred_ids, labels):
    k_pr_mean = kendalltau(pr_mean, labels[:, 0])[0]
    k_tr_mean = kendalltau(tr_mean, labels[:, 1])[0]
    k_rr_std = kendalltau(rr_std, labels[:, 2])[0]
    label_acc = recall_score(
        labels[:, 3], pred_ids, average='macro')
    label_acc = (1 - ((1 - label_acc)/(1-1/32)))
    kendall_avg = np.mean([k_pr_mean, k_tr_mean, k_rr_std])
    combined_score = np.power(
        k_rr_std * k_pr_mean * k_tr_mean * label_acc, 0.25)

    message = "Kendall TR: {} \n".format(k_tr_mean)
    message += "Kendall RR: {} \n".format(k_rr_std)
    message += "Kendall PR: {} \n".format(k_pr_mean)
    message += "ID acc: {} \n".format(label_acc)
    message += "Avg Kendall: {} \n".format(kendall_avg)
    message += "Combined: {} \n".format(combined_score)
    return message

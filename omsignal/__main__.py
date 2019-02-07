#!/usr/bin/env python3
import numpy
import torch

numpy.random.seed(42)
torch.manual_seed(42)

from omsignal.runner import run_cnn_exp, run_regression_exp


if __name__ == '__main__':
    run_regression_exp()

#!/usr/bin/env python3
import numpy
numpy.random.seed(1)
import torch
torch.manual_seed(1)

from omsignal.runner import run_cnn_exp, run_regression_exp

if __name__ == '__main__':
    run_regression_exp()

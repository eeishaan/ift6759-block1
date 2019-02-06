#!/usr/bin/env python3
import numpy
import torch

from omsignal.runner import run_cnn_exp, run_regression_exp

numpy.random.seed(42)
torch.manual_seed(42)


if __name__ == '__main__':
    run_regression_exp()

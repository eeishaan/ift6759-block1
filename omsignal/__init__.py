import logging
from logging.config import fileConfig

import numpy
import torch

from omsignal.constants import LOG_FILE_INI

# setup logging configuration
fileConfig(LOG_FILE_INI)

# fix seed for reproducability
numpy.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
  torch.cuda.manual_seed(42)

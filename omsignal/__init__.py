import logging
from logging.config import fileConfig

from omsignal.constants import CURR_DIR

# setup logging configuration
fileConfig(CURR_DIR/'logging_config.ini')

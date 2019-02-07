import logging
from logging.config import fileConfig

from omsignal.constants import LOG_FILE_INI

# setup logging configuration
fileConfig(LOG_FILE_INI)

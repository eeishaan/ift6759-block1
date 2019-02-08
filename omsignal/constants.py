'''
Module constants
'''

import os
from pathlib import Path

CURR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
LOG_FILE_INI = CURR_DIR / 'logging_config.ini'
MODEL_DIR = CURR_DIR / "saved_models"
RESULT_DIR = CURR_DIR / 'results'
GRAPH_DIR = RESULT_DIR / 'graphs'
PARAM_DIR = CURR_DIR / 'params'
ID_MAPPING = CURR_DIR / 'label_mapping.json'
DATA_ROOT_DIR = Path(
    '/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/')
TRAIN_LABELED_FILE = DATA_ROOT_DIR / 'MILA_TrainLabeledData.dat'
VALIDATION_LABELED_FILE = DATA_ROOT_DIR / 'MILA_ValidationLabeledData.dat'

import os 
from pathlib import Path


CURR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
RESULT_DIR = CURR_DIR / 'results'

DATA_ROOT_DIR = Path('omsignal/fakedata')
# ROOT_DIR: '/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/'
TRAIN_LABELED_FILE = DATA_ROOT_DIR / 'MILA_TrainLabeledData.dat'
VALIDATION_LABELED_FILE = DATA_ROOT_DIR / 'MILA_ValidationLabeledData.dat'

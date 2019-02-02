import os 
from pathlib import Path


CURR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
RESULT_DIR = CURR_DIR / 'results'
MODEL_DIR = Path("/rap/jvb-000-aa/COURS2019/etudiants/user20/ift6759")
MODEL_PATH = MODEL_DIR / 'model.pt'

# DATA_ROOT_DIR = Path('omsignal/fakedata')
DATA_ROOT_DIR = Path('/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/')
TRAIN_LABELED_FILE = DATA_ROOT_DIR / 'MILA_TrainLabeledData.dat'
VALIDATION_LABELED_FILE = DATA_ROOT_DIR / 'MILA_ValidationLabeledData.dat'

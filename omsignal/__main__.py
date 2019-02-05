#!/usr/bin/env python3
import os

import numpy as np
import torch
from torchvision.transforms import Compose

from omsignal.constants import (MODEL_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.experiments.cnn_experiment import SimpleNetExperiment
from omsignal.utils.loader import read_memfile
from omsignal.utils.transform.basic import (LabelSeparator, RemapLabels,
                                            ToTensor)
from omsignal.utils.transform.preprocessor import Preprocessor
from omsignal.utils.transform.signal import SignalSegmenter


class id_trasformer:
    def __init__(self):
        self.DictId2Class = {}
        self.DictClass2Id = {}
        self.n_Id = 0

    def create_dict_ID(self, ID_vector):
        for id in ID_vector:
            if id not in self.DictId2Class:
                self.DictId2Class[id] = self.n_Id
                self.DictClass2Id[self.n_Id] = id
                self.n_Id += 1

    def Id2Class(self, ID_vector):
        return [self.DictId2Class[id] for id in ID_vector]

    def Class2Id(self, class_vector):
        return [self.DictClass2Id[labels] for labels in class_vector]


def detect_R_peak(data, SADA_wd_size=7, FS_wd_size=12, Threshold=35):
    """
    Take a Batch of ECG data and find the location of the R Peak

    The algorithm is based on the paper:
    Online and Offline Determination of QT and PR Interval and QRS Duration in Electrocardiography
    (Bachler et al., 2012)
    The variable name and default value follow the paper

    Parameters
    ----------
    data : numpy array
        The ECG Data (batch size x lenght of the ECG recording)
    SADA_wd_size: int
        size of the moving window used in the calculation of SA and DA
    FS_wd_size: int
        size of the moving window used in the calculation of the feature signal FS
    Threshold: int
        FS is compared to the Threshold to determined if its a QRS zone. 
    """

    R_peak = []

    # Allow batch size of 1
    if len(data.size()) == 1:
        data = data.unsqueeze(0)

    D = data[:, 1:] - data[:, 0:-1]

    data = data.unsqueeze(0)
    D = D.unsqueeze(0)
    SA = F.max_pool1d(data, kernel_size=SADA_wd_size, stride=1)
    SA = SA + F.max_pool1d(-data, kernel_size=SADA_wd_size, stride=1)
    DA = F.max_pool1d(D, kernel_size=SADA_wd_size, stride=1, padding=1)
    DA = DA + F.max_pool1d(-D, kernel_size=SADA_wd_size, stride=1, padding=1)

    C = DA[:, :, 1:] * torch.pow(SA, 2)
    FS = F.max_pool1d(C, kernel_size=FS_wd_size, stride=1)
    Detect = (FS > Threshold)

    Detect = Detect.squeeze(0).cpu()
    data = data.squeeze(0).cpu()

    for ECG in range(len(data)):

        in_QRS = 0
        start_QRS = 0
        end_QRS = 0
        r_peak = np.array([])

        for tick, detect in enumerate(Detect[ECG]):

            if (in_QRS == 0) and (detect == 1):
                start_QRS = tick
                in_QRS = 1

            elif (in_QRS == 1) and (detect == 0):
                end_QRS = tick
                R_tick = torch.argmax(
                    data[ECG, start_QRS: end_QRS+SADA_wd_size+FS_wd_size]).item()
                r_peak = np.append(r_peak, R_tick + start_QRS)
                in_QRS = 0
                start_QRS = 0

        R_peak.append(r_peak)

    return R_peak


def create_template(data, R_peak, template_size=110, All_window=True):

    listoftemplate1 = []
    listoftemplate2 = []
    half_size_int = template_size//2
    listECG_id = []

    for recording in range(len(data)):
        template = []
        ECG_id = []
        # generate the template

        for i in R_peak[recording][1:-1]:
            new_heart_beat = data[recording][int(
                i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
            if len(new_heart_beat) != 110:
                print("error")
            template.append(new_heart_beat)

            if All_window:
                ECG_id.append(recording)

        if All_window == False:
            template = np.mean(template, axis=0)
            template = np.expand_dims(template, axis=0)
            ECG_id.append(recording)

        listoftemplate1 = listoftemplate1 + template
        listECG_id = listECG_id + ECG_id

    #listoftemplate = [hearth_beat for hearth_beat in template for template in listoftemplate]
    return np.array(listoftemplate1), np.array(listECG_id)


def main():
    '''
    Main function
    '''
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = MODEL_DIR / "simple_net.pt"

    datatrain = read_memfile(TRAIN_LABELED_FILE, shape=(160, 3754))
    train_data, train_labels = datatrain[:, :-4], datatrain[:, -4:]

    datavalid = read_memfile(VALIDATION_LABELED_FILE, shape=(160, 3754))
    valid_data, valid_labels = datavalid[:, :-4], datavalid[:, -4:]

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    preprocessor = Preprocessor()

    train = torch.tensor(train_data)
    valid = torch.tensor(valid_data)
    preprocess_train = preprocessor(train)
    preprocess_valid = preprocessor(valid)
    idtransformer = id_trasformer()
    idtransformer.create_dict_ID(train_labels[:, -1])
    train_labels[:, -1] = idtransformer.Id2Class(train_labels[:, -1])
    valid_labels[:, -1] = idtransformer.Id2Class(valid_labels[:, -1])

    print('Generating peaks')
    R_peak_train = detect_R_peak(preprocess_train)
    R_peak_valid = detect_R_peak(preprocess_valid)

    print('Generating templates')
    template_valid, t_valid_labels = create_template(
        preprocess_valid.cpu().numpy(),  R_peak_valid)
    template_train, t_train_labels = create_template(
        preprocess_train.cpu().numpy(), R_peak_train)

    dictECG_id_to_labels1 = {i: j for i, j in enumerate(train_labels[:, -1])}
    dictECG_id_to_labels2 = {i: j for i, j in enumerate(valid_labels[:, -1])}
    labels_train = np.array([dictECG_id_to_labels1[i] for i in t_train_labels])
    labels_valid = np.array([dictECG_id_to_labels2[i] for i in t_valid_labels])

    data_train = torch.Tensor(template_train)
    labels_train_t = torch.LongTensor(labels_train)
    trainloader = torch.utils.data.TensorDataset(data_train, labels_train_t)
    loader_train = torch.utils.data.DataLoader(
        trainloader, batch_size=128, shuffle=True)

    data_valid = torch.Tensor(template_valid)
    labels_valid_t = torch.LongTensor(labels_valid)
    validloader = torch.utils.data.TensorDataset(data_valid, labels_valid_t)
    loader_valid = torch.utils.data.DataLoader(
        validloader, batch_size=128, shuffle=False)

    simplenet_exp = SimpleNetExperiment(
        model_file,
        optimiser_params={
            'lr': 0.1
        },
        device=device
    )
    simplenet_exp.train(
        loader_train,
        epochs=3000,
        validation_dataloader=loader_valid)


if __name__ == '__main__':
    main()

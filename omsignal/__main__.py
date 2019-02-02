#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import recall_score
from torchvision.transforms import Compose

from omsignal.constants import (RESULT_DIR, TRAIN_LABELED_FILE,
                                VALIDATION_LABELED_FILE)
from omsignal.model import CNNClassifier, LSTMModel
from omsignal.utils.augmentation import SignalShift
from omsignal.utils.dim_reduction import SVDTransform, TSNETransform
from omsignal.utils.loader import (OmsignalDataset, get_dataloader,
                                   get_vector_and_labels)
from omsignal.utils.preprocessor import Preprocessor
from omsignal.utils.transform import (ClipAndFlatten, LabelSeparator,
                                      RemapLabels, SignalSegmenter, ToNumpy,
                                      ToTensor)


def task1(train_vectors, train_labels, output_folder):
    '''
    Pick three ids at random and plot their line graph
    '''
    size=3

    def _plot():
        nonlocal size
        nonlocal subset
        nonlocal label_subset
        nonlocal preprocess
        for i in range(size):
            plt.figure(i+1)
            plt.title("User ID: {} {} preprocessing"
                .format(int(label_subset[i][-1]),
                        "with" if preprocess else "without"))
            plt.plot(subset[i])
            plt.savefig(output_folder / 'vis{}_{}.png'
                .format(i+1, "p" if preprocess else "wp"))

    all_indices = np.arange(32)
    np.random.shuffle(all_indices)
    random_ids = all_indices[:size]
    random_window_indices = np.random.randint(5, size=size)
    random_indices = np.multiply(random_ids, random_window_indices)
    subset = np.take(train_vectors, random_indices, axis=0)
    label_subset = np.take(train_labels, random_indices, axis=0)
    
    preprocess = False
    _plot()
    preprocess = True
    preprocess_transform = Preprocessor().forward
    full_transform = Compose(
        [ToTensor(), preprocess_transform, ToNumpy()])
    subset = full_transform(subset)
    _plot()


def dim_reduction_task(train_vectors, train_labels, out_dir):
    '''
    Reduce dimension and plot scatter plots
    '''
    out_dim = 2
    

    def _reduce_and_plot(X, labels, transformer):
        X = transformer(X)
        color_dict = {}
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for x, label in zip(X, labels):
            user_id = label[-1]
            # use this to prevent multiple legend entries
            arg_dict = {}
            if user_id not in color_dict:
                arg_dict['label'] = user_id
                color_dict[user_id] = np.random.rand(1,3)
            arg_dict['c'] = color_dict[user_id]
            ax.scatter(x[0], x[1], **arg_dict)
        
        plt.title("Scatter plot for {}".format(transformer.name))
        plt.legend(loc=2)
        plt.savefig(out_dir / "{}.png".format(transformer.name))
        
        
    # initialize reduction transformer
    svd_transform = SVDTransform(out_dim)
    tsne_transform = TSNETransform(out_dim)

    # transform using SVD
    _reduce_and_plot(train_vectors, train_labels, svd_transform)

    # transform using t-SNE 
    _reduce_and_plot(train_vectors, train_labels, tsne_transform)
    

def train_simple_model():
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    remap_transform = RemapLabels()
    transform = Compose([
        remap_transform,
        # SignalShift(shift_len=3000),
        ToTensor()
    ])
    train_dataset = OmsignalDataset(
        TRAIN_LABELED_FILE,
        augment_transform=SignalShift(shift_len=3750),
        transform=transform)
    train_dataloader = get_dataloader(train_dataset, num_workers=0)
    validation_dataset = OmsignalDataset(
        VALIDATION_LABELED_FILE, transform=transform)
    validation_dataloader = get_dataloader(validation_dataset)
    D_in, h_1, h_2, D_out = 3750, 1024, 512, 32
    model = nn.Sequential(
        Preprocessor(),
        nn.Linear(D_in, h_1),
        nn.ReLU(),
        nn.Linear(h_1, h_2),
        nn.ReLU(),
        nn.Linear(h_2, D_out)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    epochs = 100
    for e in range(epochs):
        model.train()
        running_loss = 0
        for _, sample in enumerate(train_dataloader):
            raw_data, labels = sample[:, :-4].to(device), sample[:, -4:].to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(raw_data)
            loss = criterion(outputs, labels[:, -1])
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        for i, sample in enumerate(validation_dataloader):
            raw_data, labels = sample[:, :-4].to(device), sample[:, -4:].to(device).long()
            outputs = model(raw_data)
            _, predicted = torch.max(outputs.data, 1)
            # print(set(labels[:, -1].cpu().numpy()) - set(predicted.cpu().numpy()))
            correct += recall_score(labels[:, -1].cpu().numpy(), predicted.cpu().numpy(), average='macro')

        if e % 10 == 9:
            print('Epoch : %d Loss : %.3f ' % (e, running_loss/len(train_dataloader)))
            print('Epoch : %d Validation Score: %.3f' % (e, correct))



def train_lstm():
    device = torch.device("cuda:0" if  not torch.cuda.is_available() else "cpu")
    
    remap_transform = RemapLabels()
    transform = Compose([
        remap_transform,
        ToTensor()
    ])
    
    # initialize train dataloader
    train_dataset = OmsignalDataset(
        TRAIN_LABELED_FILE,
        augment_transform=SignalShift(shift_len=3750),
        transform=transform)
    train_dataloader = get_dataloader(train_dataset, num_workers=0, shuffle=True)
    
    # initialize validation dataloader
    validation_dataset = OmsignalDataset(
        VALIDATION_LABELED_FILE, transform=transform)
    validation_dataloader = get_dataloader(validation_dataset)
    
    D_in, h_1, D_out = 1, 512, 32

    # initialize LSTM model
    model = LSTMModel(D_in, D_out, h_1, device).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.7)

    epochs = 100
    for e in range(epochs):
        model.train()
        running_loss = 0
        for _, sample in enumerate(train_dataloader):
            raw_data, labels = sample[:, :-4].to(device), sample[:, -4:].to(device).long()
            # erase LSTM history
            model.init_hidden()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(raw_data)
            loss = criterion(outputs, labels[:, -1])
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        for _, sample in enumerate(validation_dataloader):
            raw_data, labels = sample[:, :-4].to(device), sample[:, -4:].to(device).long()
            outputs = model(raw_data)
            _, predicted = torch.max(outputs.data, 1)
            correct += recall_score(labels[:, -1].cpu().numpy(), predicted.cpu().numpy(), average='macro')

        if e % 10 == 9:
            print('Epoch : %d Loss : %.3f ' % (e, running_loss/len(train_dataloader)))
            print('Epoch : %d Validation Score: %.3f' % (e, correct))


def train_cnn():
    device = torch.device("cuda:0" if  not torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    remap_transform = RemapLabels()
    segment_size = 110
    transform = Compose([
        remap_transform,
        ToTensor(),
        LabelSeparator(),
        Preprocessor(),
        SignalSegmenter(segment_size),
    ])
    
    # initialize train dataloader
    train_dataset = OmsignalDataset(
        TRAIN_LABELED_FILE,
        transform=transform)
    clipper = ClipAndFlatten(segment_size)
    train_dataloader = get_dataloader(train_dataset, num_workers=0, shuffle=True)
        
    # initialize validation dataloader
    validation_dataset = OmsignalDataset(
        VALIDATION_LABELED_FILE,
        transform=transform)
    validation_dataloader = get_dataloader(validation_dataset, num_workers=0, shuffle=True)
    
    n_filters, kernel_size, linear_dim = 32, 5, 51

    # initialize LSTM model
    model = CNNClassifier(n_filters, kernel_size, linear_dim)\
            .to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 100
    for e in range(epochs):
        model.train()
        running_loss = 0
        for _, sample in enumerate(train_dataloader):
            data, labels = sample
            data, labels = clipper(data, labels)
            data.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            l = labels[:, -1].long()
            outputs = model(data)
            loss = criterion(outputs, l)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        print('Epoch : %d Loss : %.3f ' % (e, running_loss))
        correct = 0
        for _, sample in enumerate(validation_dataloader):
            data, labels = sample
            data, labels = clipper(data, labels)
            data.to(device)
            labels.to(device)
            l = labels[:, -1].long()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += recall_score(l.cpu().numpy(), predicted.cpu().numpy(), average='macro')

        print('Epoch : %d Validation score : %.3f' % (e, correct))
        print('--------------------------------------------------------------')


def main():
    '''
    Main function
    '''

    # fix seed for reproducability
    np.random.seed(1)
    torch.manual_seed(1)

    # turn off interactive plotting
    plt.ioff()

    train_cnn()
    # train_lstm()
    # train_simple_model()
    # separate out the labels and raw data
    # train_vectors, train_labels = get_vector_and_labels(TRAIN_LABELED_FILE)

    # valid_vectors, valid_labels = get_vector_and_labels(VALIDATION_LABELED_FILE)

    # sample = np.hstack((train_vectors, train_labels))
    # print(train_labels[:, -1])
    # x = RemapLabels()(sample)
    # print(x[: ,-1])
    # tr = SignalShift()
    # print(tr(sample).shape)
    # task1_output_folder = RESULT_DIR / 'task1' 
    # os.makedirs(task1_output_folder, exist_ok=True)
    # task1(train_vectors, train_labels, task1_output_folder)

    # scatter_plot_dir = RESULT_DIR / 'dim_reduction'
    # os.makedirs(scatter_plot_dir, exist_ok=True)
    # dim_reduction_task(train_vectors, train_labels, scatter_plot_dir)


if __name__ == '__main__':
    main()

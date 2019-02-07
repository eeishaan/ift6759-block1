#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from omsignal.utils.preprocessor import Preprocessor

class gaussian_noise(nn.Module):
    def __init__(self, std=0.1, is_relative_detach=True):
        super().__init__()
        self.std = std
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.std != 0:
            scale = self.std*x.detach() if self.is_relative_detach else self.std*x
            sampled_noise = torch.randn(x.shape).to(device)*scale
            x = x + sampled_noise
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers, device):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.device = device

        self.noise = gaussian_noise()

        self.preprocessor = Preprocessor()
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True)

        self.dropout1 = nn.Dropout(p=0.40)
        self.linear1 = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.20)
        self.linear2 = nn.Linear(int(self.hidden_dim/2), out_dim)

    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                       torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))

    def forward(self, inputs, hidden, batch_size):
        #LSTM
        inputs = self.noise(inputs)
        inputs, hidden = self.lstm(inputs, hidden)

        #MLP
        inputs = inputs.contiguous()[:, -1, :]
        inputs = self.dropout1(inputs)
        inputs = self.linear1(inputs.view(batch_size, -1))
        inputs = self.relu(inputs)
        inputs = self.dropout2(inputs)
        output = self.linear2(inputs)

        return output, hidden

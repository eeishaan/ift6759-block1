#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from omsignal.utils.preprocessor import Preprocessor


class LSTMModel(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, device):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.device = device

        self.preprocessor = Preprocessor()
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, self.out_dim)

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 20, self.hidden_dim).to(self.device),
                       torch.zeros(1, 20, self.hidden_dim).to(self.device))

    def forward(self, X):
        X = self.preprocessor(X)
        lstm_out, self.hidden = self.lstm(
            X.view(len(X), -1, 1), self.hidden)
        lstm_out = lstm_out.contiguous()[:, -1, :]
        tag_space = self.linear(lstm_out.view(20, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

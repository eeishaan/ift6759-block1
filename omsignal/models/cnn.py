#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from omsignal.utils.preprocessor import Preprocessor


class CNNClassifier(nn.Module):
    def __init__(self, n_filters=32, kernel_size=5, linear_dim=51):
        super(CNNClassifier, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.linear_dim = linear_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.n_filters,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.linear_dim*self.n_filters, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.cnn.apply(CNNClassifier.init_weights)
        self.linear.apply(CNNClassifier.init_weights)

    @classmethod
    def init_weights(cls, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(len(x), -1)
        x = self.linear(x)
        return x


class DeepCNNClassifier(nn.Module):
    def __init__(self, n_filters=32, kernel_size=3, linear_dim=51):
        super(DeepCNNClassifier, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.linear_dim = linear_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(64),

            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(128),

            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.kernel_size
            ),
            nn.ELU(),
            nn.BatchNorm1d(256),

            nn.MaxPool1d(2),

        )
        self.linear = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
        )
        self.cnn.apply(DeepCNNClassifier.init_weights)
        self.linear.apply(DeepCNNClassifier.init_weights)

    @classmethod
    def init_weights(cls, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(len(x), -1)
        x = self.linear(x)
        return x


class ShallowCNNClassifier(nn.Module):
    def __init__(self, n_filters=128, kernel_size=5, linear_dim=53):
        super(ShallowCNNClassifier, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.linear_dim = linear_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.linear_dim*self.n_filters, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.cnn.apply(ShallowCNNClassifier.init_weights)
        self.linear.apply(ShallowCNNClassifier.init_weights)

    @classmethod
    def init_weights(cls, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(len(x), -1)
        x = self.linear(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)  # 22
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)  # 18
        self.drop_conv2 = nn.Dropout2d(p=0.5)  # 4
        self.fc1 = nn.Linear(60, 50)
        self.drop_lin = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(50, 32)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool1d(self.conv1(x), 5))
        x = F.relu(F.max_pool1d(self.drop_conv2(self.conv2(x)), 5))
        x = x.view(-1, 60)
        x = F.relu(self.drop_lin(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
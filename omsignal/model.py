import torch
import torch.nn as nn
import torch.nn.functional as F
from omsignal.utils.preprocessor import Preprocessor

class LSTMModel(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.preprocessor = Preprocessor()
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
    
    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim),
                       torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, X):
        X = self.preprocessor(X)
        lstm_out, self.hidden = self.lstm(
            X.view(len(X), 1, -1), self.hidden)
        tag_space = self.linear(lstm_out.view(len(X), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
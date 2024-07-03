import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DotDict


def convert_to_dotdict(d):
    if isinstance(d, dict):
        return DotDict({k: convert_to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_to_dotdict(i) for i in d]
    else:
        return d


torch.cuda.is_available()


class Encoder(nn.Module):

    def __init__(self, params: DotDict):
        super(Encoder, self).__init__()

        self.params = params

        self.batch_size = self.params.batch_size
        self.num_stocks = int(self.params.num_stocks)
        self.num_factors = self.params.num_factors
        self.num_layers = self.params.num_layers
        self.hidden_size = self.params.hidden_size
        self.num_lags = self.params.num_lags
        self.dropout = self.params.dropout

        self.rnn = nn.LSTM(
            input_size=self.num_stocks,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        print("Hi")
        self.relu = nn.ReLU()
        self.mu = nn.Linear(self.hidden_size, self.num_factors)
        self.mu_activation = nn.ReLU()
        self.log_sigma = nn.Linear(self.hidden_size, self.num_factors)
        self.log_sigma_activation = nn.ReLU()

    def forward(self, x):
        out, (hn, cn) = self.rnn(x)
        out = self.relu(out)

        mu = self.mu(out)
        mu = self.mu_activation(mu)

        log_sigma = self.log_sigma(out)
        log_sigma = self.log_sigma_activation(log_sigma)

        return mu, log_sigma

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

class Decoder(nn.Module):

    def __init__(self, params:DotDict):
        super(Decoder, self).__init__()

        self.params = params

        self.batch_size = self.params.batch_size
        self.num_stocks = self.params.num_stocks
        self.num_factors = self.params.num_factors
        self.num_layers = self.params.num_layers
        self.hidden_size = self.params.hidden_size
        self.num_lags = self.params.num_lags
        self.dropout = self.params.dropout

        self.tanh_h0 = nn.Tanh()
        self.rnn = nn.LSTM(
                           input_size = self.num_stocks,
                           hidden_size = self.hidden_size,
                           num_layers = self.num_layers,
                           dropout = self.dropout,
                           batch_first = True
                        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.num_factors)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x, z):
        z = self.tanh_h0(z)

        out, (hn, cn)= self.rnn(x, (z, z))
        out = self.relu(out)

        out = self.fc(out)
        out = self.sigmoid(out)

        return out


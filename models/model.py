import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import TriangularMask
from models.attn import WeightAttention
from models.decoder import WeightLayer, WeightEncoder

def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    return tensor

class GMM_FNN(nn.Module):
    def __init__(self,
        hidden_units,
        dropout_rate,
        hist_len,
        pred_len,
        n_heads,
        d_ff,
        d_layers=2,
        device=torch.device('cpu')
    ):
        super(GMM_FNN, self).__init__()
        self.pred_len = pred_len
        self.device = device

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hist_len, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.last_layer = nn.Linear(hidden_units, pred_len)
        self.sig_linear = nn.Linear(hidden_units, pred_len)

        self.weight_encoder = WeightEncoder(
            [
                WeightLayer(
                    WeightAttention(n_heads, hidden_units, device),
                    hidden_units,
                    d_ff=d_ff
                )
                for l in range(d_layers)
            ]
        )

        self.weight_gen = nn.Linear(hidden_units, pred_len)
    
    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        h = F.relu(self.fc4(x))

        output = self.last_layer(h)
        sigma = F.softplus(self.sig_linear(h))

        S = self.last_layer.weight
        raw_weight = self.weight_encoder(S)
        raw_weight = self.weight_gen(raw_weight).squeeze()

        mask_gen = TriangularMask(raw_weight.size(-1), self.device)
        raw_weight = raw_weight.masked_fill(mask_gen.mask > 0, -1e9)
        final_weight = F.softmax(raw_weight, dim=1)

        return output, sigma, final_weight


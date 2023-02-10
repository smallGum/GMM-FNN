import torch.nn as nn
import torch.nn.functional as F

class WeightLayer(nn.Module):
    def __init__(self, self_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(WeightLayer, self).__init__()

        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, forward_S):
        forward_S = forward_S + self.dropout(self.self_attention(
            forward_S, forward_S, forward_S
        ))
        forward_S = self.norm1(forward_S)

        W = self.dropout(self.activation(self.linear1(forward_S)))
        W = self.dropout(self.linear2(W))

        return self.norm2(forward_S + W)

class WeightEncoder(nn.Module):
    def __init__(self, layers):
        super(WeightEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, forward_S):
        for layer in self.layers:
            backward_W = layer(forward_S)

        return backward_W
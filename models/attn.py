import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def weight_attention(query, key, value, dropout=None):
    # query: (h, pred_len, d_k)
    # R: (1, pred_len, pred_len)
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class WeightAttention(nn.Module):
    def __init__(self, h, d_model, device, dropout=0.1):
        super(WeightAttention, self).__init__()

        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.forward_linears = clones(nn.Linear(d_model, d_model), 4)
    
    def forward(self, query, key, value):
        # query, key, value: (pred_len, hidden_units)
        n_tasks = query.size(0)

        query, key, value = \
            [l(x).view(n_tasks, self.h, self.d_k).transpose(0, 1)
            for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = weight_attention(query, key, value, dropout=self.dropout)
        x = x.transpose(0, 1).contiguous().view(n_tasks, self.h * self.d_k)

        return self.linears[-1](x)
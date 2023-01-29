#%% Librairies
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Attention computation

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    output = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        output = output.masked_fill(mask == 0, -1e9)
    output = F.softmax(output, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(output, v)
    return output

#%% Multimodal Attention

class MultiModalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, lambda1=1, lambda2=1, device=device):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.q_linear = nn.Linear(d_model, d_model) 
        # For text only (indice e) 
        self.v_e_linear = nn.Linear(d_model, d_model)
        self.k_e_linear = nn.Linear(d_model, d_model)
        # For image only (indice i, parameter lambda_1)
        self.v_i_linear = nn.Linear(d_model,d_model)
        self.k_i_linear = nn.Linear(d_model,d_model)
        # From text to image (indice ei, parameter lambda_2)
        self.v_ei_linear = nn.Linear(d_model,d_model)
        self.k_ei_linear = nn.Linear(d_model,d_model)
        # From image to text (indice ie, parameter lambda_2)
        self.v_ie_linear = nn.Linear(d_model,d_model)
        self.k_ie_linear = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, v_e, k_e, v_i, k_i, v_ei, k_ei, v_ie, k_ie, mask=None):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) 
        
        # For text only 
        k_e = self.k_e_linear(k_e).view(bs, -1, self.h, self.d_k) 
        k_e = k_e.transpose(1,2)
        v_e = self.v_e_linear(v_e).view(bs, -1, self.h, self.d_k)
        v_e = v_e.transpose(1,2)
        scores_e = attention(q, k_e, v_e, self.d_k, mask, self.dropout)


        # For image only
        k_i = self.k_i_linear(k_i).view(bs, -1, self.h, self.d_k) 
        k_i = k_i.transpose(1,2)
        v_i = self.v_i_linear(v_i).view(bs, -1, self.h, self.d_k)
        v_i = v_i.transpose(1,2)
        scores_i = attention(q, k_i, v_i, self.d_k, mask, self.dropout)

        # From text to image 
        k_ei = self.k_ei_linear(k_ei).view(bs, -1, self.h, self.d_k) 
        k_ei = k_ei.transpose(1,2)
        v_ei = self.v_ei_linear(v_ei).view(bs, -1, self.h, self.d_k)
        v_ei = v_ei.transpose(1,2)
        scores_ei = attention(q, k_ei, v_ei, self.d_k, mask, self.dropout)

        # From image to text
        k_ie = self.k_ie_linear(k_ie).view(bs, -1, self.h, self.d_k) 
        k_ie = k_ie.transpose(1,2)
        v_ie = self.v_ie_linear(v_ie).view(bs, -1, self.h, self.d_k)
        v_ie = v_ie.transpose(1,2)
        scores_ie = attention(q, k_ie, v_ie, self.d_k, mask, self.dropout)

        scores = scores_e + self.lambda1 * scores_i + self.lambda2 * (scores_ei + scores_ie)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

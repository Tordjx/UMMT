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

def attention(q, k, v, d_k, mask=None, padding_mask=None, dropout=None):
    
    output = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        output = output.masked_fill(mask == 0, -1e9)
    output = F.softmax(output, dim=-1)

    if padding_mask is not None:
        soutputcores = output.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
    if dropout is not None:
        output = dropout(output)
        
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

        self.q_linear = nn.Linear(d_model,d_model)
        # For text (indice e) 
        self.v_e_linear = nn.Linear(d_model, d_model)
        self.k_e_linear = nn.Linear(d_model, d_model)
        # For image (indice i)
        self.v_i_linear = nn.Linear(d_model,d_model)
        self.k_i_linear = nn.Linear(d_model,d_model)
        # For both text and image (indice ei)
        self.v_ei_linear = nn.Linear(d_model,d_model)
        self.k_ei_linear = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, k_e, k_i, k_ei, v_e, v_i, v_ei, mask, padding_mask, image_bool):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) 
        q = q.transpose(1,2)

        # Matrices for text
        k_e = self.k_e_linear(k_e).view(bs, -1, self.h, self.d_k) 
        k_e = k_e.transpose(1,2)
        v_e = self.v_e_linear(v_e).view(bs, -1, self.h, self.d_k)
        v_e = v_e.transpose(1,2)
        scores_e = attention(q, k_e, v_e, self.d_k, mask, self.dropout) # Ici ajouter le bon mask

        # If there is only text in the input, image_bool = False
        if not(image_bool):
            concat = scores_e.transpose(1,2).contiguous().view(bs, -1, self.d_model)
            output = self.out(concat)
            return output
        else:
            # Score for image : 
            k_i = self.k_i_linear(k_i).view(bs, -1, self.h, self.d_k) 
            k_i = k_i.transpose(1,2)
            v_i = self.v_i_linear(v_i).view(bs, -1, self.h, self.d_k)
            v_i = v_i.transpose(1,2)
            scores_i = attention(q, k_i, v_i, self.d_k, mask, self.dropout)

            # Score for text and image : 
            k_ei = self.k_ei_linear(k_ei).view(bs, -1, self.h, self.d_k) 
            k_ei = k_ei.transpose(1,2)
            v_ei = self.v_ei_linear(v_ei).view(bs, -1, self.h, self.d_k)
            v_ei = v_ei.transpose(1,2)
            scores_ei = attention(q, k_ei, v_ei, self.d_k, mask, self.dropout)

            # final scores 
            scores = scores_e + self.lambda1 * scores_i + self.lambda2 * scores_ei
            concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
            output = self.out(concat)

            return output


        

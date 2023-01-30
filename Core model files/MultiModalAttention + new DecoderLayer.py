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

        self.ffn = nn.Sequential(nn.Linear(d_model,d_model),
            nn.ReLU,
            nn.Linear(d_model,2*d_model)
            )   # Quelles dimension ?? 

        self.q_linear = nn.Linear(d_model, d_model) 
        self.q_e_linear = nn.Linear(d_model,d_model)
        self.q_i_linear = nn.Linear(d_model,d_model)
        # For text only (indice e) 
        self.v_e_linear = nn.Linear(d_model, d_model)
        self.k_e_linear = nn.Linear(d_model, d_model)
        # For image only (indice i)
        self.v_i_linear = nn.Linear(d_model,d_model)
        self.k_i_linear = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, q_e, q_i, k_e, k_i, v_e, v_i, mask=None):
        bs = q.size(0)

        # A modifier !!! avec le mask
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) 
        q = q.transpose(1,2)

        q_e = self.q_e_linear(q_e).view(bs, -1, self.h, self.d_k) 
        q_e = q_e.transpose(1,2)

        q_i = self.q_i_linear(q_i).view(bs, -1, self.h, self.d_k) 
        q_i = q_i.transpose(1,2)

        # Scores for text : 
        k_e = self.k_e_linear(k_e).view(bs, -1, self.h, self.d_k) 
        k_e = k_e.transpose(1,2)
        v_e = self.v_e_linear(v_e).view(bs, -1, self.h, self.d_k)
        v_e = v_e.transpose(1,2)
        scores_e = attention(q, k_e, v_e, self.d_k, mask, self.dropout)

        # Score for image : 
        k_i = self.k_i_linear(k_i).view(bs, -1, self.h, self.d_k) 
        k_i = k_i.transpose(1,2)
        v_i = self.v_i_linear(v_i).view(bs, -1, self.h, self.d_k)
        v_i = v_i.transpose(1,2)
        scores_i = attention(q, k_i, v_i, self.d_k, mask, self.dropout)


        # Score from text to image : 
        k_ei = self.ffn(attention(q_e, k_i, v_i, self.d_k)) # Permière partie
        v_ei = self.ffn(attention(q_e, k_i, v_i, self.d_k)) # Deuxième partie 

        k_ie = self.ffn(attention(q_i, k_e, v_e, self.d_k)) # Permière partie 
        v_ie = self.ffn(attention(q_i, k_e, v_e, self.d_k)) # Deuxième partie

        scores_ei = attention(q, k_ei, v_ei, self.d_k, mask, self.dropout)
        scores_ie =  attention(q, k_ie, v_ie, self.d_k, mask, self.dropout)

        # final scores 
        scores = scores_e + self.lambda1 * scores_i + self.lambda2 * (scores_ei + scores_ie)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output









class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model,heads,dropout)
        self.attn_2 = MultiHeadAttention(d_model, n_heads, dropout, lambda1=1, lambda2=1, device=device)
        self.ff = FeedForward(d_model).cuda()
def forward(self, x, e_outputs,i_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs,i_outputs, e_outputs,i_outputs,e_outputs,i_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
    
    
    
    
    #Source: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    #Le codes Source de Torch est trop compliqué pour l'instant mais j'apprendrais -Yass

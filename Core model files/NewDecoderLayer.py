#%% Librairies
import torch.nn as nn
import torch

# Import de la classe MultimodalAttention 
from Multimodal_Attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            
#%% TransformerDecoderLayer


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads,dim_feedforward, dropout=0.1,device=device):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model,heads,dropout)
        self.attn_2 = MultiModalAttention(d_model, heads, dropout, lambda1=1, lambda2=1, device=device) # our own multiheadattention layer 
        self.ffn = nn.Sequential(nn.Linear(d_model,dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward,d_model)
            )  
    
    def forward(self, x, e_outputs, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        e_outputs = x[0]
        i_outputs = x[1]
        if i_outputs != None:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, tgt_mask)[0])
            #Here, att1 returns a tuple, the first being the result, the second being the attention weights
            x2 = self.norm_2(x)
            ei_outputs = torch.cat((e_outputs, i_outputs), 0)
            print(e_outputs.shape, x2.shape,x.shape)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, i_outputs, ei_outputs, e_outputs, i_outputs, ei_outputs, memory_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ffn(x2))
        else: #LA IL FAUT MODIFIER
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, tgt_mask)[0])
            #Here, att1 returns a tuple, the first being the result, the second being the attention weights
            x2 = self.norm_2(x)
            ei_outputs = torch.cat((e_outputs, i_outputs), 0)
            print(e_outputs.shape, x2.shape,x.shape)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, i_outputs, ei_outputs, e_outputs, i_outputs, ei_outputs, memory_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ffn(x2))
        return x

    # def forward(self, x, e_outputs,i_outputs, src_mask, tgt_mask):
    #     x2 = self.norm_1(x)
    #     x = x + self.dropout_1(self.attn_1(x2, x2, x2, tgt_mask))
    #     x2 = self.norm_2(x)
    #     ei_outputs = torch.cat((e_outputs, i_outputs), 0)
    #     x = x + self.dropout_2(self.attn_2(x2, e_outputs, i_outputs, ei_outputs, e_outputs, i_outputs, ei_outputs, src_mask))
    #     x2 = self.norm_3(x)
    #     x = x + self.dropout_3(self.ffn(x2))
    #     return x
    

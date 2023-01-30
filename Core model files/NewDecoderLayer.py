#%% Librairies
import torch.nn as nn
import torch

# Import de la classe MultimodalAttention 
from Multimodal_Attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            
#%% TransformerDecoderLayer


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.Norm(d_model)
        self.norm_2 = nn.Norm(d_model)
        self.norm_3 = nn.Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model,heads,dropout)
        self.attn_2 = MultiModalAttention(d_model, heads, dropout, lambda1=1, lambda2=1, device=device)
        self.ffn = nn.Sequential(nn.Linear(d_model,d_model),
            nn.ReLU,
            nn.Linear(d_model,d_model)
            )  # Est-ce les bonnes dimentions ? 

    def forward(self, x, e_outputs,i_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, i_outputs, e_outputs, i_outputs, e_outputs, i_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ffn(x2))
        return x
    
    
    
    
    #Source: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
    #Le codes Source de Torch est trop compliqué pour l'instant mais j'apprendrais -Yass

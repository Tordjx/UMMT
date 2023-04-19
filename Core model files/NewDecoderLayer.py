#%% Librairies
import torch.nn as nn
import torch
import csv

# Import de la classe MultimodalAttention 
from Multimodal_Attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
get_attention_csv=False
            
#%% TransformerDecoderLayer


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads,dim_feedforward, dropout=0.1,device=device, batch_first = True):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model,heads,dropout, batch_first=batch_first)
        self.attn_2 = MultiModalAttention(d_model, heads, dropout, lambda1=1, lambda2=1, device=device, batch_first=batch_first) # our own multiheadattention layer 
        self.ffn = nn.Sequential(nn.Linear(d_model,dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward,d_model)
            )  
        

        self.csv_e = open("attention_weights_e.csv", "w", newline="")
        self.writer_e = csv.writer(self.csv_e)
        
        
        self.csv_i = open("attention_weights_i.csv", "w", newline="")
        self.writer_i = csv.writer(self.csv_i)

    def forward(self, x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        if type(x) == list:  # If there is an image
            # print("cas 1 : text + image")
            text = x[0]
            i_outputs = x[1]
            x2 = self.norm_1(text)
            x=text
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)[0])
            # Here, att1 returns a tuple, the first being the result, the second being the attention weights
            x2 = self.norm_2(x)
            ei_outputs = torch.cat((memory, i_outputs), 1)
            # Here we get the two mem_masks
            memory_mask, mem_ei_mask = memory_mask
            memory_key_padding_mask, mem_ei_key_padding_mask = memory_key_padding_mask
            output,attention_weights_e,attention_weights_i=self.attn_2(x2, memory, i_outputs, ei_outputs, memory, i_outputs, ei_outputs, memory_mask, mem_ei_mask, memory_key_padding_mask, mem_ei_key_padding_mask,image_bool=True)
            x = x + self.dropout_2(output)
            if get_attention_csv:
                self.writer_e.writerows([attention_weights_e])
                self.writer_i.writerows([attention_weights_i])
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ffn(x2))
            

            
        else: # If there is only the text
            # print("case 2 : text only")
            memory_mask = memory_mask[0]
            memory_key_padding_mask = memory_key_padding_mask[0]
            text = x
            i_outputs = None
            x2 = self.norm_1(text)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)[0])
            # Here, att1 returns a tuple, the first being the result, the second being the attention weights
            x2 = self.norm_2(x)
            ei_outputs = None
            output,attention_weights_e=self.attn_2(x2, memory, i_outputs, ei_outputs, memory, i_outputs, ei_outputs, memory_mask, None, memory_key_padding_mask, None, image_bool=False)
            x = x + self.dropout_2(output)
            if get_attention_csv:
                self.writer_e.writerows([attention_weights_e])
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ffn(x2))
        return x

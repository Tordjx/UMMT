import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
import time
from typing import Tuple
from NewDecoderLayer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SI SRC=torch.rand((10,32,512)), alors d_model = 512,


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# appliquer subword nmt , nombre d'operation s = 10000 vocab 
# train_file = concatener train en anglais et train en francais

# reshape pour le resnet ou .view.reshape pour passer de 1414 
# feedforward : couche linéaire pour passer de 196x1024 aux dimensions des embeddings du texte
# mettre une option avec ou sans image où on ne fait pas 

# d_model = taille des vecteurs


class Modèle(nn.Module):
    def __init__(self,n_token, d_model, n_head, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout, activation ,padding_id,begin_id, end_id) -> None:
        super().__init__()
        self.curr_epoch=  0
        self.d_model = d_model 
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation 
        self.lr_list = []
        self.dropout = dropout
        self.n_head = n_head
        self.n_token= n_token
        self.padding_id = padding_id
        self.begin_id = begin_id
        self.end_id = end_id
        self.embedding = nn.Embedding(n_token, d_model, device=device)
        self.feedforward = nn.Linear(196,d_model,device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device = device, batch_first=True)
        decoder_layers = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device, batch_first=True) # NewDecoderLayer qui prend en compte l'image
        self.encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layers,num_decoder_layers).to(device)
        self.positional_encoder = PositionalEncoding(d_model, dropout).to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.padding_id,label_smoothing =0.1)
        self.lr = 10**(-3)# learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,betas=(0.9, 0.999), eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2, eta_min=10**(-5), last_epoch=-1)
        self.output_layer = nn.Linear(d_model, n_token).to(device)
        self.loss_list = []

    def forward(self, text_input, image_bool = False, image_input = None, mask_ei = False) : 
        src_mask = self.generate_square_subsequent_mask(self.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
        tgt_mask = self.generate_square_subsequent_mask(self.n_head*text_input.shape[0],text_input.shape[1])
        src_padding_mask  = (text_input== self.padding_id).to(device=device)
        tgt_padding_mask = (text_input== self.padding_id).to(device=device)
        # memory_mask = None
        # memory_key_padding_mask=None
        memory_mask = self.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
        memory_key_padding_mask = (text_input == self.padding_id).to(device=device)
        
        if image_bool and mask_ei:
            mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
            # mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1] + image_input.shape[1], text_input.shape[1] + image_input.shape[1]])  # Other dimension for the mem_ei_mask for test
            mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = self.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
            mem_ei_key_padding_mask = (text_input == self.padding_id).to(device=device)
            mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
        else:
            mem_ei_mask = None
            mem_ei_key_padding_mask = None

        text_encoded = self.encoder(self.positional_encoder(self.embedding(text_input)), src_mask, src_padding_mask)
        if image_bool:
            # image_input = image_input.reshape((196,1024))
            # Concatenate encoded text and image
            mem_masks = [memory_mask, mem_ei_mask]
            mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
            image_encoded = self.feedforward(image_input)
            x = [self.positional_encoder(self.embedding(text_input)), image_encoded]
            output = self.decoder(x, text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
            return self.output_layer(output)
        else:
            # Pass through the decoder
            x = text_encoded
            output = self.decoder(self.positional_encoder(self.embedding(text_input)),x , tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
            # return self.activation(self.output_layer(output))
            return self.output_layer(output)

# masque rectangle
    # def generate_square_subsequent_mask(self,sz_1=40,sz_2=35):
    #     return torch.triu(torch.full((sz_1,sz_2 ), float('-inf'), device=device), diagonal=1)

    def generate_square_subsequent_mask(self,a,b) -> Tensor:
        return torch.triu(torch.full((a,b,b), True, device=device,dtype = bool), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
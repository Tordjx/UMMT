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
d_model = 1024
n_head = 4 
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward= 100
dropout = 0.1
activation = nn.Softmax

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


class Modèle (nn.Module):
    def __init__(self,n_token,d_model,n_head, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout, activation ) -> None:
        super().__init__()
        self.d_model = d_model 
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation 
        self.dropout = dropout
        self.n_head = n_head
        self.n_token= n_token

        self.embedding = nn.Embedding(n_token, d_model,device=  device)
        self.feedforward = nn.Linear(d_model,d_model,device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device = device)
        decoder_layers = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device) # NewDecoderLayer qui prend en compte l'image
        self.encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layers,num_decoder_layers).to(device)
        self.positional_encoder = PositionalEncoding(d_model, dropout).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # Implementer la controlable attention : copier coller le code source et ajuster , mettre lambda 2 quand meme

    def forward(self, text_input, image_bool = False , image_input = None) : 
        
        # Encode Text
        text_encoded = self.encoder(self.positional_encoder(self.embedding(text_input)))
        mask = self.generate_square_subsequent_mask(text_input.shape[0])
        if image_bool:
            image_input =image_input.reshape((196,1024))
            # Concatenate encoded text and image
            image_encoded =self.feedforward(image_input)
            
            output = self.decoder(self.positional_encoder(self.embedding(text_input)),text_encoded, image_encoded ,mask)
        
            return output
        else:
            image_encoded = None
            # Pass through the decoder
            
            output = self.decoder(self.positional_encoder(self.embedding(text_input)),text_encoded,image_encoded ,mask)
        
            return output


    


    def generate_square_subsequent_mask(self,sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

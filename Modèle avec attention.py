import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
import time
from typing import Tuple

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


class Modèle(nn.Module):
    def __init__(self, n_token, d_model, n_head, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation):
        super().__init__()
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.n_head = n_head

        self.embedding = nn.Embedding(n_token, d_model, device=device)
        self.feedforward = nn.Linear(d_model, d_model, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, device=device)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.self_attn_layer= nn.MultiheadAttention(d_model, n_head, dropout=dropout)  # Add self-attention layer
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device=device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.masked_self_attn_layer = MaskedSelfAttention(d_model, n_head, dim_feedforward, dropout)
        self.multi_modal_attn_layer = MultiModalAttention(d_model, n_head, dim_feedforward, dropout)

        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def forward(self, input, src_mask, bool_image):
        if bool_image:
            resnet = input[1].reshape((196, 1024))

            embedded = self.embedding(input[0])
            pos_enc = self.positional_encoder(embedded)
            
            self_attn_output, _ = self.self_attention(pos_enc, pos_enc, pos_enc)
            encoder_output = self.encoder(self_attn_output)
            
            masked_self_attn_output = self.masked_self_attn_layer(pos_enc, src_mask)
            input_decoder = self.multi_modal_attn_layer(masked_self_attn_output, resnet, encoder_output)
            
            decoder_output = self.decoder(input_decoder)
            
            #structure génerale pour me mettre les idées au clair. Je suis en train de recoder les layers from scratch puisqu'il y a déja de l'attention dedans 
            
           

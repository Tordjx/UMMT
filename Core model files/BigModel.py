import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
import time
from typing import Tuple
from NewDecoderLayer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class Modèle(nn.Module):
    def __init__(self,n_token_A, n_token_B, d_model, n_head, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout, activation, padding_id_A, padding_id_B ) -> None:
        super().__init__()
        self.loss_list = []
        self.d_model = d_model 
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation 
        self.dropout = dropout
        self.n_head = n_head
        self.n_token_A= n_token_A
        self.n_token_B= n_token_B
        self.padding_id_A=padding_id_A
        self.padding_id_B=padding_id_B
        self.embedding_A = nn.Embedding(n_token_A, d_model, device=device)
        self.feedforward_A = nn.Linear(d_model,d_model,device=device)
        encoder_layers_A = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device = device, batch_first=True)
        decoder_layers_A = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device, batch_first=True) # NewDecoderLayer qui prend en compte l'image
        self.encoder_A = nn.TransformerEncoder(encoder_layers_A,num_encoder_layers).to(device)
        self.decoder_A = nn.TransformerDecoder(decoder_layers_A,num_decoder_layers).to(device)
        self.positional_encoder = PositionalEncoding(d_model, dropout).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 10**(-3)# learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.output_layer_A = nn.Linear(d_model, n_token_A).to(device)
        self.embedding_B = nn.Embedding(n_token_B, d_model, device=device)
        self.feedforward_B = nn.Linear(d_model,d_model,device=device)
        encoder_layers_B = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device = device, batch_first=True)
        decoder_layers_B = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device, batch_first=True) # NewDecoderLayer qui prend en compte l'image
        self.encoder_B = nn.TransformerEncoder(encoder_layers_B,num_encoder_layers).to(device)
        self.decoder_B = nn.TransformerDecoder(decoder_layers_B,num_decoder_layers).to(device)
        self.output_layer_B = nn.Linear(d_model, n_token_B).to(device)
        
    def forward(self,forward_type,data_source, text_input, image_bool=False, image_input = None, mask_ei = True):
        if forward_type not in ['Auto encoding', 'Cycle','Differentiable cycle'] or data_source not in ['A','B']:
            raise Exception('Not a valid forward or data source')
        else :
            if data_source == "A" :
                PADDING = self.padding_id_A
            else : 
                PADDING = self.padding_id_B
            src_mask=self.generate_mask(text_input)
            tgt_mask=self.generate_mask(text_input)
            src_padding_mask= self.generate_padding_mask(text_input, PADDING)
            tgt_padding_mask =  self.generate_padding_mask(text_input, PADDING)
            memory_mask = torch.triu(torch.full((text_input.shape[0],text_input.shape[1],text_input.shape[1]), float('-inf'), device=device), diagonal=1)
            memory_padding_mask = self.generate_padding_mask(text_input, PADDING)
            if mask_ei and image_bool: 
                memory_ei_mask , memory_ei_padding_mask= self.generate_ei_mask(text_input,PADDING,image_input)
            else :
                memory_ei_mask , memory_ei_padding_mask=None,None
                ########################################################
            if forward_type == 'Auto encoding' :
                if data_source == "A" :
                    text_encoded = self.encoder_A(self.positional_encoder(self.embedding_A(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_A(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_A(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_A(output)
                else : 
                    text_encoded = self.encoder_B(self.positional_encoder(self.embedding_B(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_B(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_B(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_B(output)
            ########################################################
            elif forward_type == "Cycle" :
                if data_source == "A" :
                    text_encoded = self.encoder_A(self.positional_encoder(self.embedding_A(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_A(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_B(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_B(output)
                else : 
                    text_encoded = self.encoder_B(self.positional_encoder(self.embedding_B(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_B(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_A(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_A(output)
            ########################################################
            else :#Differentiable cycle 
                if data_source == "A" :
                    ############A INPUT
                    text_encoded = self.encoder_A(self.positional_encoder(self.embedding_A(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_A(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                    else : 
                        x = text_encoded
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                    with torch.no_grad():
                        text_input = torch.argmax(self.output_layer_B(output),dim = 2)
                    if data_source == "A" :
                        PADDING = self.padding_id_B
                    else : 
                        PADDING = self.padding_id_A
                    src_mask=self.generate_mask(text_input)
                    tgt_mask=self.generate_mask(text_input)
                    src_padding_mask= self.generate_padding_mask(text_input, PADDING)
                    tgt_padding_mask =  self.generate_padding_mask(text_input, PADDING)
                    memory_mask = torch.triu(torch.full((text_input.shape[0],text_input.shape[1],text_input.shape[1]), float('-inf'), device=device), diagonal=1)
                    memory_padding_mask = self.generate_padding_mask(text_input, PADDING)
                    if mask_ei and image_bool : 
                        memory_ei_mask , memory_ei_padding_mask= self.generate_ei_mask(text_input,PADDING,image_input)
                    else :
                        memory_ei_mask , memory_ei_padding_mask=None,None
                    text_encoded = self.encoder_B(self.positional_encoder(output), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_B(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_A(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_A(output)
                ######B INPUT
                else : 
                    text_encoded = self.encoder_B(self.positional_encoder(self.embedding_B(text_input)), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_B(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                    else : 
                        x = text_encoded
                        output = self.decoder_A(x, self.positional_encoder(self.embedding_B(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                    with torch.no_grad():
                        text_input = torch.argmax(self.output_layer_A(output),dim = 2)
                    if data_source == "A" :
                        PADDING = self.padding_id_B
                    else : 
                        PADDING = self.padding_id_A
                    src_mask=self.generate_mask(text_input)
                    tgt_mask=self.generate_mask(text_input)
                    src_padding_mask= self.generate_padding_mask(text_input, PADDING)
                    tgt_padding_mask =  self.generate_padding_mask(text_input, PADDING)
                    memory_mask = torch.triu(torch.full((text_input.shape[0],text_input.shape[1],text_input.shape[1]), float('-inf'), device=device), diagonal=1)
                    memory_padding_mask = self.generate_padding_mask(text_input, PADDING)
                    text_encoded = self.encoder_A(self.positional_encoder(output), src_mask, src_padding_mask)
                    if image_bool : 
                        mem_masks = [memory_mask, memory_ei_mask]
                        mem_padding_masks = [memory_padding_mask, memory_ei_padding_mask]
                        image_encoded = self.feedforward_A(image_input)
                        x = [text_encoded, image_encoded]
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
                        return self.output_layer_B(output)
                    else : 
                        x = text_encoded
                        output = self.decoder_B(x, self.positional_encoder(self.embedding_A(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_padding_mask])
                        return self.output_layer_B(output)
            
    
    def generate_padding_mask(self, text_input,padding_id):
        return (text_input== padding_id).to(device=device)
    def generate_ei_mask(self,text_input,padding_id,image_input):
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = torch.triu(torch.full((text_input.shape[0],text_input.shape[1],text_input.shape[1]), float('-inf'), device=device), diagonal=1)
        mem_ei_key_padding_mask = self.generate_padding_mask(text_input, padding_id)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
        return mem_ei_mask , mem_ei_key_padding_mask
    def generate_mask(self,text_input) : 
        a = self.n_head*text_input.shape[0]
        b=text_input.shape[1]
        return torch.triu(torch.full((a,b,b), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
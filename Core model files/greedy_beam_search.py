#%% Librairies 

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

#%% Greedy 

def greedy_decode(model_A,model_B,text_input, image_input = None, image_bool = False):

    EOS_IDX= model_B.end_id
    SOS_IDX= model_B.begin_id
    PAD_IDX= model_B.begin_id
    max_len = 97

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    tgt_padding_mask = (text_input==  model_A.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)

    memory = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
    ys=torch.cat((torch.ones(batch_size ,1,dtype = torch.int).fill_(model_A.begin_id),torch.ones(batch_size ,96,dtype = torch.int).fill_(model_A.padding_id)),dim =1)
        
    if image_bool: 
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
    
    for i in range(max_len-1):
        if image_bool:
            x = [model_A.positional_encoder(model_A.embedding(ys)), image_encoded]
            out = model_B.decoder(x,memory, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
        else:
            x = model_A.positional_encoder(model_A.embedding(ys))
            out = model_B.decoder(x,memory, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])

        # out = out.transpose(0, 1) est ce utile ?
        prob = model_B.output_layer(out)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        if next_word == EOS_IDX:
            break

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

    return ys


#%% Beam search 

def beam_search(model_A,model_B,text_input, image_input = None, image_bool = False, beam_size=3):

    EOS_IDX= model_B.end_id
    SOS_IDX= model_B.begin_id
    PAD_IDX= model_B.begin_id
    max_len = 97

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    tgt_padding_mask = (text_input==  model_A.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)

    memory = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
    ys=torch.cat((torch.ones(batch_size ,1,dtype = torch.int).fill_(model_A.begin_id),torch.ones(batch_size ,96,dtype = torch.int).fill_(model_A.padding_id)),dim =1)
        
    if image_bool: 
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
    
    for i in range(max_len-1):
        if image_bool:
            x = [model_A.positional_encoder(model_A.embedding(ys)), image_encoded]
            out = model_B.decoder(x,memory, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
        else:
            x = model_A.positional_encoder(model_A.embedding(ys))
            out = model_B.decoder(x,memory, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])

        logs_prob = torch.log_softmax(model_B.outputlayer(out), dim=1)
        # Here we can penalize the longest sentences ... 
        print(logs_prob.shape)
        break


    #     # out = out.transpose(0, 1) est ce utile ?
    #     prob = model_B.output_layer(out)
    #     _, next_word = torch.max(prob, dim=1)
    #     next_word = next_word.item()

    #     if next_word == EOS_IDX:
    #         break

    #     ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

    # return ys



#%% Tests : 

from Modele_decodeur_maison import Modèle
from Pipeline import get_train_data_nouveau

# Texts
tokenized_fr,tokenized_en, vocab_fr,vocab_en = get_train_data_nouveau(batch_size)
#Data non batchés
n_token_fr = len(vocab_fr.keys())
n_token_en = len(vocab_en.keys())

inv_map_en = {v: k for k, v in vocab_en.items()}
inv_map_fr = {v: k for k, v in vocab_fr.items()}

n_head =4 
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024
dropout = 0.1
activation = nn.Softmax(dim=2)
embedding_dim = 512

model_fr = Modèle(n_token_fr,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_fr["TOKEN_VIDE"],vocab_fr["DEBUT_DE_PHRASE"],vocab_fr["FIN_DE_PHRASE"]).to(device)
model_en = Modèle(n_token_en,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_en["TOKEN_VIDE"],vocab_en["DEBUT_DE_PHRASE"],vocab_en["FIN_DE_PHRASE"]).to(device)
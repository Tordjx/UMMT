#%% Librairies 

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

#%% Greedy 
 
def CCF_greedy(model_A,model_B,text_input, image_input = None, image_bool = False): 
    max_len = 97

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)

    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)

    text_input = torch.cat((torch.ones(batch_size ,1,dtype = torch.int).fill_(model_B.begin_id),torch.ones(batch_size ,96,dtype = torch.int).fill_(model_B.padding_id)),dim =1)
    
    for i in range(max_len-1):

        tgt_mask = model_B.generate_square_subsequent_mask(model_B.n_head*text_input.shape[0],text_input.shape[1])
        tgt_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
        memory_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        if image_bool:
            mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
            mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
            mem_ei_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
            mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)

        if image_bool :  
            x = [model_B.positional_encoder(model_B.embedding(text_input)), image_encoded]
            output = model_B.decoder(x,text_encoded, None , mem_masks , None, mem_padding_masks)
        else:
            x = text_encoded
            output = model_B.decoder(model_B.positional_encoder(model_B.embedding(text_input)),x, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
        
        # Greedy 
        prob =  model_B.output_layer(output)
        next_words = torch.argmax(prob, dim=2)[:,i]
        text_input[:,i] = next_words

    return model_B.output_layer(output)


#%% Beam search 

def CCF_beam_search(model_A,model_B,text_input, image_input = None, image_bool = False, beam_size=3):
    max_len = 97

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)

    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
    
    decoder_input = torch.cat((torch.ones(batch_size ,1,dtype = torch.int).fill_(model_B.begin_id),torch.ones(batch_size ,96,dtype = torch.int).fill_(model_B.padding_id)),dim =1)

    for i in range(max_len-1):

        tgt_mask = model_B.generate_square_subsequent_mask(model_B.n_head*decoder_input.shape[0],decoder_input.shape[1])
        tgt_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
        memory_mask = model_A.generate_square_subsequent_mask(decoder_input.shape[0],decoder_input.shape[1])
        memory_key_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
        if image_bool:
            mem_ei_mask = torch.zeros([decoder_input.shape[0], decoder_input.shape[1], decoder_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
            mem_ei_mask[:,0:decoder_input.shape[1], 0:decoder_input.shape[1]] = model_A.generate_square_subsequent_mask(decoder_input.shape[0],decoder_input.shape[1]).to(device=device)
            mem_ei_key_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
            mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([decoder_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)

        if image_bool :  
            x = [model_B.positional_encoder(model_B.embedding(decoder_input)), image_encoded]
            output = model_B.decoder(x,text_encoded, None , mem_masks , None, mem_padding_masks)
        else:
            x = text_encoded
            output = model_B.decoder(model_B.positional_encoder(model_B.embedding(decoder_input)),x, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])

        # Beam search 
        logs_prob = torch.log_softmax(model_B.output_layer(output), dim=1)
        # Here we can penalize the longest sentences ... 
        print(logs_prob)
        break



#%% Data for tests : 

from Modele_decodeur_maison import Modèle
from Pipeline import get_train_data_nouveau, batchify

# Texts
tokenized_fr,tokenized_en, vocab_fr,vocab_en = get_train_data_nouveau(batch_size)
#Data non batchés
n_token_fr = len(vocab_fr.keys())
n_token_en = len(vocab_en.keys())

n_head =4 
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024
dropout = 0.1
activation = nn.Softmax(dim=2)
embedding_dim = 512

model_fr = Modèle(n_token_fr,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_fr["TOKEN_VIDE"],vocab_fr["DEBUT_DE_PHRASE"],vocab_fr["FIN_DE_PHRASE"]).to(device)
model_en = Modèle(n_token_en,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_en["TOKEN_VIDE"],vocab_en["DEBUT_DE_PHRASE"],vocab_en["FIN_DE_PHRASE"]).to(device)

val_features = np.load("C:/Users/lucas/Desktop/val-resnet50-res4frelu.npy")
val_features = torch.from_numpy(val_features)
train_data_en = tokenized_en
batched_data = batchify(train_data_en,batch_size,False)
data = batched_data

#%% Tests 

CCF_beam_search(model_fr,model_en, data[0])
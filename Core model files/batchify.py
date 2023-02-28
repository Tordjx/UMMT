#%% libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

#%% Texts functions

def get_vocab() : 
    fichier_vocab_fr = open('vocab.fr')
    fichier_vocab_en = open('vocab.en')
    vocab_en = [line.split()[0] for line in fichier_vocab_en if len(line.split()) == 2]
    vocab_en = dict((y,x) for (x,y) in enumerate(vocab_en))
    vocab_fr = [line.split()[0] for line in fichier_vocab_fr if len(line.split()) == 2]
    vocab_fr = dict((y,x) for (x,y) in enumerate(vocab_fr))
    fichier_vocab_en.close()
    fichier_vocab_fr.close()
    return vocab_en, vocab_fr

def get_train_data_text():
    vocab_en,vocab_fr = get_vocab()
    fichier_train_fr = open('train.BPE.fr')
    fichier_train_en = open('train.BPE.en')
    train_data_fr = [ligne.strip().split(" ") for ligne in fichier_train_fr ]
    train_data_en = [ligne.strip().split(" ") for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()

    for ligne in train_data_en: 
        for mot in ligne: 
            if mot not in vocab_en: 
                vocab_en[mot] = len(vocab_en.keys())

    for ligne in train_data_fr: 
        for mot in ligne: 
            if mot not in vocab_fr: 
                vocab_fr[mot] = len(vocab_fr.keys())

    embedded_fr = [torch.tensor([vocab_fr[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_fr]
    embedded_en = [torch.tensor([vocab_en[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_en]

    # A quoi sert cette étape ? On perd l'indo de la séparation des phrases ? 
    train_final_fr = torch.cat(embedded_fr)
    train_final_en = torch.cat(embedded_en)

    # return [embedded_en, embedded_fr]
    return [train_final_en,train_final_fr]

#%% Tests :

def batchify(data: Tensor,device, bsz: int = 20) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 40 

train_final_en,train_final_fr = get_train_data_text()

train_data_fr = batchify(train_final_fr, device, batch_size)




#%% Image functions

def get_train_data_image():
    return np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")






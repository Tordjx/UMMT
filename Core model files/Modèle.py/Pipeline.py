#%%Batchifier
import torch
from torch import Tensor
from typing import Tuple

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
def get_train_data():
    vocab_en,vocab_fr = get_vocab()
    fichier_train_fr = open('train.BPE.fr')
    fichier_train_en = open('train.BPE.en')
    train_data_fr = [ligne.strip().split(" ") for ligne in fichier_train_fr ]
    train_data_en = [ligne.strip().split(" ") for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()


    #ATTENTION : DEMANDER IUN AVIS POUR CES DEUX BOUCLES QUI AJOUTENT AU VOCAB CE QUI MANQUE, CEST A DIRE '&@@' et ';@@' a cause des caracteres html
    for ligne in train_data_en : 
        for mot in ligne : 
            if mot not in vocab_en : 
                vocab_en[mot] = len(vocab_en.keys())

    for ligne in train_data_fr : 
        for mot in ligne : 
            if mot not in vocab_fr : 
                vocab_fr[mot] = len(vocab_fr.keys())
    embedded_fr = [torch.tensor([vocab_fr[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_fr]
    embedded_en = [torch.tensor([vocab_en[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_en]

    train_final_fr = torch.cat(embedded_fr)
    train_final_en = torch.cat(embedded_en)
    return [train_final_en,train_final_fr]
#A cet endroit, on a un flat tensor de données comme dans le tuto transformers
#Les lignes suivantes viennent directement du tuto transformer
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


# eval_batch_size = 10

# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)
#ICI CEST LE BATCHIFIER DU AUTO ENCODING!!!!!!!!!!!!S
bptt = 35
def get_batch(source: Tensor, i: int,device) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].reshape(-1) CAS GENERAL
    return data.to(device), target.to(device)



#%%Batchifier
import torch
from torch import Tensor
from typing import Tuple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_train_data_nouveau():
    vocab_en,vocab_fr = get_vocab()
    fichier_train_fr = open('train.BPE.fr')
    fichier_train_en = open('train.BPE.en')
    train_data_fr = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+['FIN_DE_PHRASE'] for ligne in fichier_train_fr ]
    train_data_en = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+["FIN_DE_PHRASE"] for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()
    # longueur_max = max(max([len(x) for x in train_data_fr]),max( [len(x) for x in train_data_fr]))
    # Arrondissons à 100
    longueur_max=100
    train_data_fr = [[phrase[i] if i < len(phrase) else "TOKEN_VIDE" for i in range (longueur_max)] for phrase in train_data_fr]
    train_data_en = [[phrase[i] if i < len(phrase) else "TOKEN_VIDE" for i in range (longueur_max)] for phrase in train_data_en]


    #ATTENTION : DEMANDER IUN AVIS POUR CES DEUX BOUCLES QUI AJOUTENT AU VOCAB CE QUI MANQUE, CEST A DIRE '&@@' et ';@@' a cause des caracteres html
    for ligne in train_data_en: 
        for mot in ligne : 
            if mot not in vocab_en: 
                print(mot)
                vocab_en[mot] = len(vocab_en.keys())

    for ligne in train_data_fr: 
        for mot in ligne: 
            if mot not in vocab_fr: 
                print(mot)
                vocab_fr[mot] = len(vocab_fr.keys())

    tokenized_fr = [torch.tensor([vocab_fr[x]  for x in ligne ], dtype= torch.long).to(device) for ligne in train_data_fr]
    tokenized_en = [torch.tensor([vocab_en[x]  for x in ligne ], dtype= torch.long).to(device) for ligne in train_data_en]
#va falloir aussi return les nouveaux vocab

    return [tokenized_fr,tokenized_en, vocab_fr,vocab_en]

def batchify(data: Tensor,device, bsz: int = 10) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(1) // bsz
    data = data.view(data.size(1) , seq_len, bsz)
    return data.to(device)


# eval_batch_size = 10

# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)
#ICI CEST LE BATCHIFIER DU AUTO ENCODING!!!!!!!!!!!!S

bptt = 10

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
    target = source[i:i+seq_len].reshape(-1)

    return data.to(device), target.to(device)

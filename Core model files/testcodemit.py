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
    train_data_fr = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+['FIN_DE_PHRASE'] for ligne in fichier_train_fr ]
    train_data_en = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+["FIN_DE_PHRASE"] for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()
    longueur_max = max(max([len(x) for x in train_data_fr]),max( [len(x) for x in train_data_fr]))
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

    tokenized_fr = [torch.tensor([vocab_fr[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_fr]
    tokenized_en = [torch.tensor([vocab_en[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_en]
#va falloir aussi return les nouveaux vocab

    return [tokenized_fr,tokenized_en, vocab_fr,vocab_en]

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

bptt = 40

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



# # %%
# from torchtext.datasets import WikiText2
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import dataset
# train_iter = WikiText2(split='train')
# tokenizer = get_tokenizer('basic_english')
# vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
# vocab.set_default_index(vocab['<unk>'])

# def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
#     """Converts raw text into a flat Tensor."""
#     data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
#     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# # train_iter was "consumed" by the process of building the vocab,
# # so we have to create it again
# train_iter, val_iter, test_iter = WikiText2()
# train_data = data_process(train_iter)
# val_data = data_process(val_iter)
# test_data = data_process(test_iter)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def batchify(data: Tensor,device, bsz: int) -> Tensor:
#     """Divides the data into bsz separate sequences, removing extra elements
#     that wouldn't cleanly fit.

#     Args:
#         data: Tensor, shape [N]
#         bsz: int, batch size

#     Returns:
#         Tensor of shape [N // bsz, bsz]
#     """
#     seq_len = data.size(0) // bsz
#     data = data[:seq_len * bsz]
#     data = data.view(bsz, seq_len).t().contiguous()
#     return data.to(device)

# batch_size = 20
# eval_batch_size = 10
# train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)


# %%

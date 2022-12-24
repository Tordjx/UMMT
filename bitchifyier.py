#%%
import torch
from torch import Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
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

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
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
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

#%%
data, targets = get_batch(train_data, 1)
seq_len = data.size(0)

#%%
fichier_vocab_fr = open('vocab.fr')
fichier_vocab_en = open('vocab.en')
vocab_en = [line.split()[0] for line in fichier_vocab_en if len(line.split()) == 2]
vocab_en = dict((y,x) for (x,y) in enumerate(vocab_en))
vocab_fr = [line.split()[0] for line in fichier_vocab_fr if len(line.split()) == 2]
vocab_fr = dict((y,x) for (x,y) in enumerate(vocab_fr))

def cherche_vocab(mot, vocab) : 
    resultat = []
    i = 0
    while i< len(mot):
        plus_grand = 0
        for j in range(len(mot)) :
            if mot[i:i+j] in vocab :
                séparé = mot[i:i+j]
                plus_grand = j 
        i+=plus_grand
        resultat.append(séparé)
        

    return resultat
print(cherche_vocab('absolument', vocab_fr))
def tokenizer(vocab, texte) :
    resultat=[]
    for ligne in texte.split("\n") :
        for mot in ligne.split(" "):
            resultat.append(cherche_vocab(mot,vocab))
        resultat.append('\n') 
    return resultat

# tokenizer (vocab_fr , 'je suis un boulanger, je mange du pain et de la teub', "@@")
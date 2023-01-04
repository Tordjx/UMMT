#%%
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


class Modèle (nn.Module):
    def __init__(self,n_token,d_model,n_head, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout, activation ) -> None:
        super().__init__()
        self.d_model = d_model 
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation 
        self.n_head = n_head

        self.embedding = nn.Embedding(n_token, d_model,device=  device)
        self.feedforward = nn.Linear(d_model,d_model,device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device=device)
        decoder_layers= nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device)
        self.encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layers,num_decoder_layers)
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # Implementer la controlable attention : copier coller le code source et ajuster , mettre lambda 2 quand meme

    def forward(self, input,src_mask,bool_image) : 
    #L'encoder prend en entrée obligatoire une phrase embedded et le positional encoding
    #Le decoder prend en entrée l'output de l'encoder, l'output du resnet, un masked self attention et le positionnal encoding
        if bool_image :
            resnet = input[1].reshape((196,1024))

            embedded = self.embedding(input[0])
            pos_enc = self.positional_encoder(embedded)
            input_decoder = self.encoder(pos_enc) #c'est la phrase
            input_decoder = self.feedforward(resnet) * input_decoder
            output = self.decoder (input_decoder, src_mask(pos_enc))
            return activation(output)
        else : 
            embedded = self.embedding(input)
            pos_enc = self.positional_encoder(embedded)
            input_decoder = self.encoder(pos_enc) #c'est la phrase
            output = self.decoder (input_decoder , src_mask)
            return activation(output)

    # def controllable_attention(self, lambda_1 , lambda_2) :
    #     Terme_1 = 
    #     Terme 2 = 0 
    #     Terme 3 = 0 
    #     if lambda_1 : 
    #         Terme_2 = 
    #     if lambda_2 : 
    #         Terme_3 = 
    #     return Terme_1+Terme_2+Terme_3


    def generate_square_subsequent_mask(self,sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



                
    def eval(self, n_iter,eval_data) : 
        self.eval() #Turn on evaluation mode
        total_loss = 0 
        with torch.no_grad():
            for i in range(len(eval_data)):
                data = eval_data[i]
                seq_len = data.size(0)
                output = self(data)
                total_loss += seq_len * self.criterion(output, data)
        return total_loss /(len(eval_data) -1 )

    def traduire(self, input):
        output = self(input)
        return output_to_sentence(output)



#%%Batchifier
#%%
fichier_vocab_fr = open('vocab.fr')
fichier_vocab_en = open('vocab.en')
vocab_en = [line.split()[0] for line in fichier_vocab_en if len(line.split()) == 2]
vocab_en = dict((y,x) for (x,y) in enumerate(vocab_en))
vocab_fr = [line.split()[0] for line in fichier_vocab_fr if len(line.split()) == 2]
vocab_fr = dict((y,x) for (x,y) in enumerate(vocab_fr))
fichier_vocab_en.close()
fichier_vocab_fr.close()
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
#A cet endroit, on a un flat tensor de données comme dans le tuto transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Les lignes suivantes viennent directement du tuto transformer
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
# eval_batch_size = 10
train_data_fr = batchify(train_final_fr, batch_size)  # shape [seq_len, batch_size]
train_data_en = batchify(train_final_en, batch_size)  # shape [seq_len, batch_size]

# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

bptt = 35
#ICI CEST LE BATCHIFIER DU AUTO ENCODING!!!!!!!!!!!!
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
    target = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].reshape(-1) CAS GENERAL
    return data, target



#%%
n_token_fr = len(vocab_fr.keys())
n_token_en = len(vocab_en.keys())
Modèle_fr = Modèle(n_token_fr,d_model , n_head, num_encoder_layers , num_decoder_layers , dim_feedforward, dropout, activation).to(device)
def train_auto_encoding(model,train_data):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        print(data.device,targets.device,  src_mask.device)
        output = model(data, src_mask,False)
        loss = model.criterion(output.view(-1, model.ntokens),targets)

        model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        model.optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = model.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = np.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


train_auto_encoding(Modèle_fr,train_data_fr)
# %%

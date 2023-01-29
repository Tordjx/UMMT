#%% Librairies
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy


#%% Embedding
class Embedder(nn.Module): # nn.Module est la classe de base de chaque nn. 

    def __init__(self, vocab_size, d_model):
        super().__init__()  # super() permet l'héritage entre nn.Module et Embedder
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


#%% Positional encoding 

class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # Creation de la matrice pos. encoding
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)  # Ajoute 1 à la dimension du tensor 
        self.register_buffer('pe', pe) # Pas vraiment compris l'utilité


    def forward(self, x):  # x est la matrice d'entrée avec les mots embedded
        x = x * math.sqrt(self.d_model)  # Calcul qui permet de rendre la partie embedding plus importante 
        seq_len = x.size(1)
        # Pas vraiment certains de la différence entre ces deux lignes : 
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False)
        # x = x + self.pe[:,:seq_len] 
        return x

#%% Masks

# size = 
nopeak_mask = np.triu(np.ones((1, 10, 10)), k=1).astype('uint8')
# nopeak_mask = torch.from_numpy(subsequent_mask) == 0.5

#%% Attention 

def attention(q, k, v, d_k, mask=None):
    
    output = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        output = output.masked_fill(mask == 0, -1e9)
    output = F.softmax(output, dim=-1)
    
    # Certains articles rajoutent une étape de dropout. A quoi sert cette étape ??
        
    output = torch.matmul(output, v)
    return output


#%% Multihead attention

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model) # nn.Linear applique la transformation linaire xA^t + b à l'entrée
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  # randomly zeroes some of the elements of the input
        self.out = nn.Linear(d_model, d_model)
    

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) #  .view change la dimension du tensor
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # Ici on applique la transformation lineaire 
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # Puis on 
       
        k = k.transpose(1,2)   # Pour avoir les bonnes dimensions
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#%% Feed forward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


#%% Normalisation

class Norm(nn.Module):

    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm




#%% Encoder and encoder layer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    

#%% Decoder et decoder layer 

class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


#%% Transformers

class Transformer(nn.Module):

    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

#%% Test 


x = torch.randn(4, 4)
print(x)
print(x.size())
y = x.view(16)
print(y.size())
print(y)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(z.size())
print(z)

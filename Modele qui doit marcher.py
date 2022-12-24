#%%
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SI SRC=torch.rand((10,32,512)), alors d_model = 512,
d_model = 1024
n_head = 4 
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward= 100
dropout = 0.1
activation = nn.Softmax
fichier_en = open('vocab.en')
n_token_en = len(fichier_en.readlines())
fichier_fr = open('vocab.fr')
n_token_fr = len(fichier_fr.readlines())
fichier_en.close()
fichier_fr.close()
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
    def __init__(self,n_token,d_model,n_head, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout, activation,device ) -> None:
        super().__init__()
        self.d_model = d_model 
        self.num_encoder_layers= num_encoder_layers
        self.num_decoder_layers= num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation 
        self.n_head = n_head
        self.device = device
        self.embedding = nn.Embedding(n_token, d_model)
        self.feedforward = nn.Linear(d_model,d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,device=self.device)
        decoder_layers= nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, device = device)
        self.encoder = nn.TransformerEncoder(encoder_layers,num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layers,num_decoder_layers)
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # Implementer la controlable attention : copier coller le code source et ajuster , mettre lambda 2 quand meme

    def forward(self, input) : 
    #L'encoder prend en entrée obligatoire une phrase embedded et le positional encoding
    #Le decoder prend en entrée l'output de l'encoder, l'output du resnet, un masked self attention et le positionnal encoding
        if input[1] != None :
            resnet = input[1].reshape((196,1024))

            embedded = self.embedding(input[0])
            pos_enc = self.positional_encoder(embedded)
            input_decoder = self.encoder(pos_enc) #c'est la phrase
            input_decoder = self.feedforward(resnet) * input_decoder
            output = self.decoder (input_decoder, self.generate_square_subsequent_mask(pos_enc))
            return activation(output)
        else : 
            embedded = self.embedding(input[0])
            pos_enc = self.positional_encoder(embedded)
            input_decoder = self.encoder(pos_enc) #c'est la phrase
            output = self.decoder (input_decoder , self.generate_square_subsequent_mask(pos_enc))
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
        return torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def train(self, n_iter,train_data) : 

        # a chaque batch on tire soit l'un soit l'autre des loss
        self.train() #Turn on train mode
        total_loss = 0
        log_interval = 200
        start_time =time.time()
        for i in range(len(train_data)) : 
            output = self(train_data[i])
            loss = self.criterion(output,train_data[i])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            if i%log_interval==0 :
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print("Current loss " + str(cur_loss) + "ms_per_batch " + str(ms_per_batch))
                total_loss = 0
                start_time = time.time()
                
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


Modèle(n_token_fr,d_model , n_head, num_encoder_layers , num_decoder_layers , dim_feedforward, dropout, activation , device)





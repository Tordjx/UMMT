#%%
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder,TransformerDecoderLayer,Softmax

# Transformer encoder : 4* self attention et feedforword
# Transformer decoder : 4* (masked self attention et controlable multi modal attention et feed forward) et softmax



class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                  dropout: float = 0.5, nlayers  = 2 ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers,Softmax)
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Linear(d_model, ntoken)
        self.d_hid = d_hid
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src: Tensor, src_mask : Tensor) -> Tensor:
        output = self.encoder(src) * np.sqrt(self.d_model)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output,src_mask)
        output = self.transformer_decoder(output,src_mask)
        output = self.decoder(output)
        return output
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


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
#%%
data_path = "C:/Users/valen/Documents/ENSAE/STATAPP"
fichier_train = data_path+'/multi30k-dataset/data/task1/tok/train.lc.norm.tok'
fichier_eval = data_path+'/multi30k-dataset/data/task1/tok/val.lc.norm.tok'


SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def readLang(lang1):
    print("Reading lines...")
    # Read the file and split into lines
    train_data = open(fichier_train+'.%s' % (lang1), encoding='utf-8').\
        read().strip().split('\n')
    eval_data = open(fichier_eval+'.%s' % (lang1), encoding='utf-8').\
        read().strip().split('\n')
    return Lang(lang1) , train_data,eval_data
    
def prepareData(lang1):
    input_lang, train_data,eval_data = readLang(lang1)
    print("Read %s sentence " %(len(train_data)+len(eval_data)))
    print("Counting words...")
    for line in train_data:
        input_lang.addSentence(line)
    for line in eval_data:
        input_lang.addSentence(line)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, train_data,eval_data


langue, train_data,eval_data = prepareData('en')
print(np.random.choice(train_data), np.random.choice(eval_data))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ntokens = langue.n_words  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nhead = 4  # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

#%%
import time
def train_auto_encoding(langue, model : nn.Module ) -> None : 
    model.train() #Turn on train mode
    total_loss = 0
    log_interval = 200
    start_time =time.time()
    for i in range(len(train_data)) : 
        data = tensorFromSentence(langue,train_data[i])
        print(train_data[i])
        src_mask = model.generate_square_subsequent_mask(len(train_data[i]))
        output = model(data, src_mask)
        loss = criterion(output,data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if i%log_interval==0 :
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print("Current loss " + str(cur_loss) + "ms_per_batch " + str(ms_per_batch))
            total_loss = 0
            start_time = time.time()

def evaluate(model : nn.Module, eval_data: Tensor) -> float:
    model.eval() #Turn on evaluation mode
    total_loss = 0 
    with torch.no_grad():
        for i in range(len(eval_data)):
            data = eval_data[i]
            seq_len = data.size(0)
            output = model (data)
            total_loss += seq_len * criterion(output, data)
    return total_loss /(len(eval_data) -1 )

#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

#%%

train_auto_encoding(langue, model)
# %%

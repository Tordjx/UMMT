#%%
from Modele_decodeur_maison import *
from Pipeline import * 

from Trainer import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100 

# Images
# images = np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")

# Texts
train_final_en,train_final_fr = get_train_data()
print(train_final_fr.shape)
train_data_fr = batchify(train_final_fr, device,batch_size)  
train_data_en = batchify(train_final_en, device, batch_size) 

vocab_en,vocab_fr = get_vocab()
n_token_fr = len(vocab_fr.keys())
n_token_en = len(vocab_en.keys())
print(n_token_en)

n_head = 4 
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 100
dropout = 0.1
activation = nn.Softmax

model_fr = Modèle(n_token_fr,batch_size,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation)

model_en = Modèle(n_token_en,batch_size,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation)


#%%
data,target = get_batch(train_data_en,0,device)

model_en(data)
# train_auto_encoding(model_fr,train_data_fr)

# %%
# ACTUELLEMENT, LE SOUCI EST LE BATCHIFYER. NOTRE INPUT A POUR SHAPE [40,40], DONC LOUTPUT DU EMBEDDING EST [40,40,d_model]
# POUR AVOIR LA DIMENSION CORRECTE, IL FAUDRAIT QUON AIT UNE INPUT DE TAILLE [40]

#%%
from Modele_decodeur_maison import *
from Pipeline import * 

from Trainer import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 40 

# Images
# images = np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")

# Texts
train_final_en,train_final_fr = get_train_data()
print(train_final_fr.shape)
train_data_fr = batchify(train_final_fr, device,batch_size)  
train_data_en = batchify(train_final_en, device, batch_size) 

vocab_en,vocab_fr = get_vocab()
n_token_fr = len(vocab_fr.keys())+3
n_token_en = len(vocab_en.keys())+3
print(n_token_en)

n_head = 4 
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 100
dropout = 0.1
activation = nn.Softmax
embedding_dim = 200

model_fr = Modèle(n_token_fr,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation)

model_en = Modèle(n_token_en,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation)


#%%
data,target = get_batch(train_data_fr,0,device)

model_fr(data)


#%%
train_auto_encoding(model_fr,train_data_fr)

# #%%
# text_input = data
# print(text_input.shape)
# print(model_en.embedding(text_input).shape)
# print(model_en.positional_encoder(model_en.embedding(text_input)).shape)
# print(model_en.encoder(model_en.positional_encoder(model_en.embedding(text_input))).shape)
# print(model_en.encoder(model_en.positional_encoder(model_en.embedding(text_input))).shape)
# mask = model_en.generate_square_subsequent_mask(text_input.shape[0]) # square mask 
# x = model_en.encoder(model_en.positional_encoder(model_en.embedding(text_input)))
# output = model_en.decoder(x, model_en.positional_encoder(model_en.embedding(text_input)), mask)
# print(output.shape)
# print(model_en.output_layer(output).shape)

#%%
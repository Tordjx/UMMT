#%%
# from Pipeline import * 
from Pipeline import *
import time
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bptt = 10
epoch = 1
def train_auto_encoding(model,train_data):

    loss_list=[]
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 10
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) 
    for i in range (len(train_data)):

        # data, target = get_batch(train_data, i,device)
        data= batchify(train_data[i],device,10)
        target = train_data[i]
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        # print(data.device,target.device,  src_mask.device)
        output = model(data)
        loss = model.criterion(output.view(-1, model.n_token),target)

        model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        model.optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
        if i % log_interval == 0 and i > 0:
            lr = model.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = np.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return loss_list

def auto_encoding_train(model,train_data):

    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    # data, target = get_batch(train_data, i,device)
    data= batchify(train_data,device,10)
    target = train_data
    seq_len = data.size(0)
    if seq_len != bptt:  # only on last batch
        src_mask = src_mask[:seq_len, :seq_len]
    # print(data.device,target.device,  src_mask.device)
    output = model(data)
    loss = model.criterion(output.view(-1, model.n_token),target)
    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model.optimizer.step()
    return loss.item()


# %%

def cycle_consistent_forward(model_A,model_B,text_input, image_input = None, image_bool = False) : 
    # Encode Text
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)))
    mask = model_A.generate_square_subsequent_mask(text_input.shape[0]) # square mask 
    if image_bool:
        image_input =image_input.reshape((196,1024))
        # Concatenate encoded text and image
        image_encoded =model_A.feedforward(image_input)
        encoded = torch.cat([text_encoded, image_encoded], dim=1)
    else:
        encoded = text_encoded
    # Pass through the decoder
    output = model_B.decoder(encoded,model_A.positional_encoder(model_A.embedding(text_input)),mask)
    return model_B.output_layer(output)

    
# def cycle_consistency_train(model_fr, model_en,train_data_fr,train_data_en):
#     loss_list = []
#     model_fr.train()
#     model_en.train()
#     total_loss = 0.
#     log_interval = 200
#     start_time = time.time()
#         #Avec proba 1/2 on commence avec fr/EN
#     if np.random.rand()<1/2 : 
#         train_data= train_data_en
#         model_A = model_en
#         model_B = model_fr
#     else : 
#         train_data= train_data_fr
#         model_A = model_fr
#         model_B = model_en
#     num_batches = len(train_data) 
#     for i in range(len(train_data)):
#         data= batchify(train_data[i],device,10)
#         target = train_data[i]
#         seq_len = data.size(0)
#         if seq_len != bptt:  # only on last batch
#             src_mask = src_mask[:seq_len, :seq_len]
#         output = cycle_consistent_forward(model_B,model_A, torch.argmax(cycle_consistent_forward(model_A,model_B, data),dim = 2))
#         loss = model_A.criterion(output.view(-1, model_A.n_token),target)

#         model_A.optimizer.zero_grad()
#         model_B.optimizer.zero_grad()
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
#         torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)
#         model_A.optimizer.step()
#         model_B.optimizer.step()

#         total_loss += loss.item()
#         loss_list.append(loss.item())
#         if i % log_interval == 0 and i > 0:
#             lr = model_A.scheduler.get_last_lr()[0]
#             ms_per_batch = (time.time() - start_time) * 1000 / log_interval
#             cur_loss = total_loss / log_interval
#             ppl = np.exp(cur_loss)
#             print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
#                   f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
#                   f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
#             total_loss = 0
#             start_time = time.time()
#     return loss_list


def cycle_consistency_train(model_A, model_B,train_data):

        data= batchify(train_data,device,10)
        target = train_data
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = cycle_consistent_forward(model_B,model_A, torch.argmax(cycle_consistent_forward(model_A,model_B, data),dim = 2))
        loss = model_A.criterion(output.view(-1, model_A.n_token),target)
        model_A.optimizer.zero_grad()
        model_B.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)
        model_A.optimizer.step()
        model_B.optimizer.step()
        return loss.item()

import matplotlib.pyplot as plt
def mixed_train(model_fr,model_en,train_data_fr,train_data_en,n_iter):
    loss_list = []
    model_fr.train()
    model_en.train()
    log_interval = 200
    total_loss = 0
    start_time = time.time()
    for i_iter in range(n_iter):
        for i in range(len(train_data_fr)):
            if np.random.rand()<1/2 : #Cycle consistency
                if np.random.rand()<1/2 : 
                    train_data= train_data_en[i]
                    model_A = model_en
                    model_B = model_fr
                else : 
                    train_data= train_data_fr[i]
                    model_A = model_fr
                    model_B = model_en
                loss = cycle_consistency_train(model_A,model_B,train_data)
            else: #Auto encoding
                if np.random.rand()<1/2 : #English
                    train_data= train_data_en[i]
                    model_A = model_en
                else : #French
                    train_data= train_data_fr[i]
                    model_A = model_fr
                loss = auto_encoding_train(model_A,train_data)
            loss_list.append(loss)
            total_loss+=loss
            if i%log_interval == 0 and i !=0 : 
                print("Iteration : " + str(i_iter) + " data numéro : "+str(i)+" en "+ str(int(1000*(time.time()-start_time)/log_interval)) + " ms par itération, moyenne loss "+ str(total_loss/200)) 
                total_loss = 0
                start_time = time.time()
        plt.plot(loss_list)
    
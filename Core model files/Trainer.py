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

def auto_encoding_train(model,train_data, image_bool):

    if image_bool : 
        data, feature = train_data
    else : 
        data,target = train_data
    # data, target = get_batch(train_data, i,device)
    # print(data.device,target.device,  src_mask.device)
    if image_bool : 
        output = model(data,True,feature)
        loss = model.criterion(output.mT,data)
    else : 
        output = model(data)
        loss = model.criterion(output.mT,target)
    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model.optimizer.step()
    return loss.item()



def cycle_consistent_forward(model_A,model_B,text_input, image_input = None, image_bool = False) : 
    # Encode Text
    
    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input== 6574).to(device=device)
    tgt_padding_mask = (text_input== 6574).to(device=device)
    # src_padding_mask = None
    # tgt_padding_mask=None
    memory_mask = None
    memory_key_padding_mask =None
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
    if image_bool:
        # Concatenate encoded text and image
        image_encoded =model_A.feedforward(image_input)
        x = [text_encoded, image_encoded]
    else:
        x = text_encoded
    # Pass through the decoder
    output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , memory_mask , tgt_padding_mask, memory_key_padding_mask)
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


def cycle_consistency_train(model_A, model_B,train_data,image_bool=False):
        
        if image_bool : 
            data, features = train_data
        else :
            data,target = train_data


        if image_bool : 
            
            output = cycle_consistent_forward(model_B,model_A, torch.argmax(cycle_consistent_forward(model_A,model_B, data, features, image_bool),dim = 2), features, image_bool)
            loss = model_A.criterion(output.mT,data)
        else :
            output = cycle_consistent_forward(model_B,model_A, torch.argmax(cycle_consistent_forward(model_A,model_B, data),dim = 2))
            loss = model_A.criterion(output.mT,target)
        model_A.optimizer.zero_grad()
        model_B.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)
        
        model_A.optimizer.step()
        model_B.optimizer.step()
        return loss.item()

import matplotlib.pyplot as plt
def mixed_train(model_fr,model_en,train_data_fr,train_data_en,n_iter,batch_size, image_bool = False):
    loss_list = []
    model_fr.train()
    model_en.train()
    log_interval = 50
    total_loss = 0
    start_time = time.time()
    for i_iter in range(n_iter):
        if image_bool : 
            N = len(train_data_fr[0])
        else : 
            N = len(train_data_fr[0])
        for i in range(N):
            if np.random.rand()<1/2 : #Cycle consistency
                
                if np.random.rand()<1/2 : 
                    if image_bool : 
                        train_data= get_batch(train_data_en,i,image_bool)
                        
                    else : 
                        train_data= get_batch(train_data_en,i)
                    model_A = model_en
                    model_B = model_fr
                else : 
                    if image_bool : 
                        train_data= get_batch(train_data_fr,i,image_bool)
                        
                    else : 
                        train_data= get_batch(train_data_fr,i)
                    model_A = model_fr
                    model_B = model_en
                loss = cycle_consistency_train(model_A,model_B,train_data,image_bool)
            else: #Auto encoding
                
                if np.random.rand()<1/2 : #English
                    if image_bool : 
                        train_data= get_batch(train_data_en,i,image_bool)
                        
                    else : 
                        train_data= get_batch(train_data_en,i)
                    model_A = model_en
                else : #French
                    if image_bool : 
                        train_data= get_batch(train_data_fr,i,image_bool)
                        
                    else : 
                        train_data= get_batch(train_data_fr,i)
                    model_A = model_fr
                loss = auto_encoding_train(model_A,train_data,image_bool)
            loss_list.append(loss)
            total_loss+=loss
            
            
            
            if (i%log_interval == 40 and i !=0) or i == N-1 : 
                print("Iteration : " + str(i_iter) + " batch numéro : "+str(i)+" en "+ str(int(1000*(time.time()-start_time)/log_interval)) + " ms par itération, moyenne loss "+ str(total_loss/log_interval)) 
                total_loss = 0
                start_time = time.time()
        plt.plot(loss_list)
    
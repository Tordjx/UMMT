#%%
# from Pipeline import * 
from Pipeline import *
import time
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bptt = 10
epoch = 1
# torch.autograd.set_detect_anomaly(True)
def auto_encoding_train(model,train_data, image_bool):
    if image_bool : 
        data, feature = train_data
    else : 
        data,target = train_data
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
    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    tgt_padding_mask = (text_input==  model_A.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
    print("encoded text")
    print(text_encoded)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
        x = [text_encoded, image_encoded]
        output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
        print("post decoder")
        print(output)
    else:
        x = text_encoded
        output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
        
    return model_B.output_layer(output)

def differentiable_cycle_forward(model_A,model_B,text_input, image_input = None, image_bool = False, mask_ei = False):
    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input==  model_A.padding_id).to(device=device)
    tgt_padding_mask = (text_input==  model_A.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
    if image_bool and mask_ei:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    else:
        mem_ei_mask = None
        mem_ei_key_padding_mask = None
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
        x = [text_encoded, image_encoded]
        output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
    else:
        x = text_encoded
        output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
    #Intermediate result to have the new masks
    with torch.no_grad():
        text_input = torch.argmax(model_B.output_layer(output),dim = 2)
    #Compute new masks with augmented data
    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
    src_padding_mask  = (text_input==  model_B.padding_id).to(device=device)
    tgt_padding_mask = (text_input== model_B.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input == model_B.padding_id).to(device=device)
    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input == model_B.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    text_encoded = model_B.encoder(model_B.positional_encoder(output),src_mask,src_padding_mask)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
        x = [text_encoded, image_encoded]
        output = model_A.decoder(x,model_B.positional_encoder(model_B.embedding(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
    else:
        x = text_encoded
        output = model_A.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
    return model_A.output_layer(output)

def differentiable_cycle_consistency_train(model_A, model_B,train_data,image_bool=False):
        if image_bool : 
            data, features = train_data
        else :
            data,target = train_data
        if image_bool : 
            output = differentiable_cycle_forward(model_A,model_B, data, features, image_bool)
            loss = model_A.criterion(output.mT,data)
        else :
            output = differentiable_cycle_forward(model_A,model_B, data)
            loss = model_A.criterion(output.mT,target)
        model_A.optimizer.zero_grad()
        model_B.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)    
        model_A.optimizer.step()
        model_B.optimizer.step()
        return loss.item()

def cycle_consistency_train(model_A, model_B,train_data,image_bool=False):
        if image_bool : 
            data, features = train_data
        else :
            data,target = train_data
        if image_bool : 
            with torch.no_grad():
                first_output = cycle_consistent_forward(model_A,model_B, data, features, image_bool)
                print("before argmax")
                print(first_output)
                first_output = torch.argmax(first_output,dim = 2)
            print('first output')
            print(first_output)
            output = cycle_consistent_forward(model_B,model_A, first_output, features, image_bool)
            print('output')
            print(output)
        else :
            with torch.no_grad() : 
                first_output = torch.argmax(cycle_consistent_forward(model_A,model_B, data),dim = 2)
            output = cycle_consistent_forward(model_B,model_A, first_output)
        loss_A = model_A.criterion(output.mT,data)
        loss_B = model_B.criterion(output.mT,data)
        model_A.optimizer.zero_grad()
        model_B.optimizer.zero_grad()
        loss_A.backward(retain_graph=True)
        loss_B.backward()
        torch.nn.utils.clip_grad_norm_(model_A.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model_B.parameters(), 5)
        model_A.optimizer.step()
        model_B.optimizer.step()
        
        return (loss_A.item()+loss_B.item())/2
import matplotlib.pyplot as plt
def mixed_train(model_fr,model_en,train_data_fr,train_data_en,n_iter,batch_size, image_bool = False,repartition = [1/2,1/2]):
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
            N = len(train_data_fr)
        for i in range(N):
            U = np.random.rand()
            V = np.random.rand()
            if U<1/2 : #ENGLISH DATA
                if image_bool : 
                    train_data= get_batch(train_data_en,i,image_bool)
                else : 
                    train_data= get_batch(train_data_en,i)
                model_A = model_en
                model_B = model_fr
            else : #FRENCH DATA
                if image_bool : 
                        train_data= get_batch(train_data_fr,i,image_bool)
                else : 
                    train_data= get_batch(train_data_fr,i)
                model_A = model_fr
                model_B = model_en
            if V < repartition[0]  :#AUTO ENCODING
                loss = auto_encoding_train(model_A,train_data,image_bool)
                model_A.loss_list.append(loss)
            elif V < repartition[1]  :#CYCLE CONSISTENT
                loss = cycle_consistency_train(model_A,model_B,train_data,image_bool)
                model_A.loss_list.append(loss)
                model_B.loss_list.append(loss)
            else : #DIFFERENTIABLE CYCLE
                loss = differentiable_cycle_consistency_train(model_A,model_B,train_data,image_bool)
            loss_list.append(loss)
            print(loss)
            total_loss+=loss
            if (i%log_interval == 40 and i !=0) or i == N-1 : 
                print("Iteration : " + str(i_iter) + " batch numéro : "+str(i)+" en "+ str(int(1000*(time.time()-start_time)/log_interval)) + " ms par itération, moyenne loss "+ str(total_loss/log_interval)) 
                total_loss = 0
                start_time = time.time()
        plt.plot(loss_list)
    
    

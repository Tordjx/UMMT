from Pipeline import *
import time
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bptt = 50
epoch = 1
torch.autograd.set_detect_anomaly(True)
#CONVENTION A = FR B = EN
import matplotlib.pyplot as plt
def mixed_train(model,train_data_fr,train_data_en,n_iter,batch_size, repartition, image_bool = False):
    model.train()
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
                    text_data,features= get_batch(train_data_en,i,image_bool)
                else : 
                    text_data,_= get_batch(train_data_en,i)
                data_source = 'B'
            else : #FRENCH DATA
                if image_bool : 
                    text_data,features= get_batch(train_data_fr,i,image_bool)
                else : 
                    text_data,_= get_batch(train_data_fr,i)
                data_source = 'A'
            if V < repartition[0]  :#AUTO ENCODING
                if image_bool : 
                    output = model('Auto encoding', data_source, text_data,True,features)
                else : 
                    output = model('Auto encoding', data_source, text_data)
            elif V<repartition[1]  :#CYCLE CONSISTENT
                if image_bool : 
                    with torch.no_grad():
                        output = torch.argmax( model('Cycle', data_source, text_data,True,features),dim = 2)
                    if data_source == 'B' :
                        data_source = 'A'
                    else :
                        data_source = 'B'
                    output = model('Cycle', data_source, output,True,features)
                else : 
                    with torch.no_grad():
                        output = torch.argmax( model('Cycle', data_source, text_data),dim = 2)
                    if data_source == 'B' :
                        data_source = 'A'
                    else :
                        data_source = 'B'
                    output = model('Cycle', data_source, output)
            else : #DIFFERENTIABLE CYCLE
                if image_bool : 
                    output = model('Differentiable cycle', data_source, text_data,True,features)
                else : 
                    output = model('Differentiable cycle', data_source, text_data)
            loss = model.criterion(output.mT,text_data)
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            model.optimizer.step()
            model.loss_list.append(loss.item())
            print(loss.item())
            total_loss+=loss
            if (i%log_interval == 40 and i !=0) or i == N-1 : 
                print("Iteration : " + str(i_iter) + " batch numéro : "+str(i)+" en "+ str(int(1000*(time.time()-start_time)/log_interval)) + " ms par itération, moyenne loss "+ str(total_loss/log_interval)) 
                total_loss = 0
                start_time = time.time()
        plt.plot(loss_list)
#%%
from Pipeline import * 
import time
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_auto_encoding(model,train_data):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, target = get_batch(train_data, i,device)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        print(data.device,target.device,  src_mask.device)
        output = model(data)
        # loss = model.criterion(output.view(-1, model.n_token),target)
        loss = model.criterion(output,target)
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



# %%

def cycle_consistent_forward(model_A,model_B,text_input, image_input = None, image_bool = False) : 
    # Encode Text
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)))
    if image_bool:
        image_input =image_input.reshape((196,1024))
        # Concatenate encoded text and image
        image_encoded =model_A.feedforward(image_input)
        encoded = torch.cat([text_encoded, image_encoded], dim=1)
    else:
        encoded = text_encoded
    # Pass through the decoder
    output = model_B.decoder(model_B.positional_encoder(model_B.embedding(text_input)),encoded)
    return output

    
def cycle_consistency_train(model_fr, model_en,train_data_fr,train_data_en):
    model_fr.train()
    model_en.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
        #Avec proba 1/2 on commence avec fr/EN
    if np.random.rand()<1/2 : 
        train_data= train_data_en
        model_A = model_en
        model_B = model_fr
    else : 
        train_data= train_data_fr
        model_A = model_fr
        model_B = model_en
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, target = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        print(data.device,target.device,  src_mask.device)
        output = cycle_consistent_forward(model_B,model_A, cycle_consistent_forward(model_A,model_B, data))

        loss = model_A.criterion(output.view(-1, model_A.n_token),target)

        model_A.optimizer.zero_grad()
        model_B.optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)
        model_A.optimizer.step()
        model_B.optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = model_A.scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = np.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from greedy_beam_search import *
from Pipeline import *
import numpy as np

def tensor_to_sentence(output,inv_dic):
    result = [inv_dic[int(x)] for x in output]
    sentence = ""
    for word in result : 
        if word == "DEBUT_DE_PHRASE" or word == "TOKEN_VIDE" :
            pass
        elif '@@' in word: 
            sentence+=word[:-2]
        elif word == "FIN_DE_PHRASE" :
            break 
        else :
            sentence+=word +" "
    return sentence


def give_tokens(output, padding_id, end_id ) : #takes output of greedy search tensor of size [bsz,seqlen,n_token]. returns the tokens, with no padding before end of sentence token
    #todoso, just need to take the 2nd outpuuts
    values, indices = torch.kthvalue(output, 2 , dim = 2)
    sentences = torch.argmax(output, dim  = 2 ) # size bsz, seqlen
    #we modify this sentences tensor
    for i in range(sentences.size(0)):#batch
        authorize_padding = False
        for j in range(sentences.size(1)):#sentence
            if sentences[i][j] == end_id:
                authorize_padding = True
            elif sentences[i][j] == padding_id and not authorize_padding : 
                sentences[i][j] = indices[i][j]
    return sentences


def traduit(mode,model_A,model_B,src, inv_map_src,image_bool,tgt,inv_map_tgt,j):
    model_A.eval()
    model_B.eval()
    if image_bool : 
        data,features= src
    else :
        data= src
    # 
    with open(model_A.prefix+"logs.txt",'a') as logs :
        if image_bool :
            if mode == 'greedy':
                output = give_tokens(CCF_greedy(model_A,model_B,data, features, True),model_B.padding_id,model_B.end_id)[j]
            else : 
                output = CCF_beam_search(model_A, model_B, data, 3, features, True)[j]
        else :
            if mode == 'greedy':
                output = give_tokens(CCF_greedy(model_A,model_B,data, None, False),model_B.padding_id,model_B.end_id)[j]
            else : 
                output = CCF_beam_search(model_A, model_B, data, 3, None, False)[j]
        logs.write("\nBleu score\n")
        bleu = evalue_bleu(output.view(-1),inv_map_tgt,tgt.view(-1))
        logs.write(str(bleu))
        logs.write("\nMeteor score\n")
        meteor = evalue_meteor(output.view(-1),inv_map_tgt,tgt.view(-1))
        logs.write(str(meteor))
        logs.write("\nBonne traduction\n")
        logs.write(str(tensor_to_sentence(tgt.view(-1),inv_map_tgt)))
        logs.write("\nOutput\n")
        logs.write(str(output))
        logs.write("\nAuto encoding\n")
        # print(tensor_to_sentence(torch.argmax(model_A(data,True,features),dim = 2).view(-1),inv_map_src))
        if image_bool :
            logs.write(str(tensor_to_sentence(torch.argmax(model_A(data,True,features),dim = 2)[j].view(-1),inv_map_src)))
        else :
            logs.write(str(tensor_to_sentence(torch.argmax(model_A(data,False),dim = 2)[j].view(-1),inv_map_src)))
        logs.close()
    return tensor_to_sentence(output.view(-1),inv_map_tgt),bleu,meteor

def evalue_bleu(output,inv_dic_tgt, tgt):
    result = [inv_dic_tgt[int(x)] for x in output if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    target = [inv_dic_tgt[int(x)] for x in tgt if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    return sentence_bleu([target], result)
def evalue_meteor(output,inv_dic_tgt, tgt):
    result = [inv_dic_tgt[int(x)] for x in output if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    target = [inv_dic_tgt[int(x)] for x in tgt if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    return meteor_score([target], result)

def donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool):
    batched_data_en,batched_data_fr=batchify([val_data_en,val_data_fr],batch_size,image_bool)
    if image_bool:
        src,features = batched_data_en
        tgt,_ = batched_data_fr
        return src[i],features[i],tgt[i]
    else :
        src,tgt = batched_data_en,batched_data_fr
        return src[i],tgt[i]

def evaluation(mode,val_data_en,val_data_fr,batch_size,model_en,model_fr,inv_map_en,inv_map_fr,image_bool):
    if image_bool :
        tokenized_val_en = val_data_en[0]
    else : 
        tokenized_val_en = val_data_en
    i = np.random.randint(len(tokenized_val_en)//batch_size)
    j = np.random.randint(batch_size)
    if image_bool: 
        src,features,tgt = donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool)
        features = features.to(device,dtype=torch.float32)
        data = [src,features]
    else :
        src,tgt = donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool)
        data = src
    
    trad,bleu,meteor = traduit(mode,model_en,model_fr,data, inv_map_en,image_bool,tgt[j],inv_map_fr,j) 
    with open(model_en.prefix+"logs.txt",'a') as logs :
        logs.write("\nPhrase à traduire : \n" + tensor_to_sentence(src[j],inv_map_en)+ "\nPhrase traduite : \n"+ trad)
        logs.close()
    return bleu, meteor

# import torchtext
import pandas as pd
def dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size) :
    model_fr.eval()
    model_en.eval()
    batched_data_en,batched_data_fr=batchify([val_data_en,val_data_fr],batch_size,image_bool,conservative=True,permute = False)
    if image_bool:
        src,features = batched_data_en
        tgt,_ = batched_data_fr
    else :
        src,tgt = batched_data_en,batched_data_fr
    traductions_en_fr = []
    references_en = []
    traductions_fr_en = []
    references_fr = []
    traductions_en_fr_txt_only = []
    traductions_fr_en_txt_only = []
    for batch in range(len(src)):
        print(batch)
        if image_bool :
            src[batch],features[batch],tgt[batch] = src[batch].to(device),features[batch].to(device),tgt[batch].to(device)
            traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],features[batch],image_bool) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_en_fr.append([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_fr[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_fr.append([inv_map_fr[tgt[batch][i][j].item()] for j in range(tgt[batch].shape[1]) if inv_map_fr[tgt[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],features[batch],image_bool) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_fr_en.append([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_en[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_en.append([inv_map_en[src[batch][i][j].item()] for j in range(src[batch].shape[1]) if inv_map_en[src[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],None,False) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_en_fr_txt_only.append([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_fr[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],None,False) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_fr_en_txt_only.append([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_en[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
        else :
            src[batch] ,tgt[batch]= src[batch].to(device),tgt[batch].to(device)
            traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],None,image_bool) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_en_fr.append([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_fr[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_fr.append([inv_map_fr[tgt[batch][i][j].item()] for j in range(tgt[batch].shape[1]) if inv_map_fr[tgt[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],None,image_bool) ,dim = 2)
            for i in range(traduction.shape[0]):
                traductions_fr_en.append([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1]) if inv_map_en[traduction[i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_en.append([inv_map_en[src[batch][i][j].item()] for j in range(src[batch].shape[1]) if inv_map_en[src[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])

    if image_bool:
        data = {"traductions_en_fr":traductions_en_fr,"references_fr":references_fr,"traductions_fr_en":traductions_fr_en,"references_en":references_en,"traductions_en_fr_txt_only":traductions_en_fr_txt_only,"traductions_fr_en_txt_only":traductions_fr_en_txt_only}
    else :
        data = {"traductions_en_fr":traductions_en_fr,"references_fr":references_fr,"traductions_fr_en":traductions_fr_en,"references_en":references_en}

    df = pd.DataFrame(data)
    return df.loc()[:val_data_fr[0].shape[0]-1]
def bleu(row,langue_src):
    if langue_src == "en":
        candidates= list(row["traductions_en_fr"])
        references = [list(row["references_fr"])]
        return sentence_bleu(references, candidates,smoothing_function=SmoothingFunction().method4)
    else : 
        candidates= list(row["traductions_fr_en"])
        references = [list(row["references_en"])]
        return sentence_bleu(references, candidates,smoothing_function=SmoothingFunction().method4)
def bleu_txt_only(row,langue_src):
    if langue_src == "en":
        candidates= list(row["traductions_en_fr_txt_only"])
        references = [list(row["references_fr"])]
        return sentence_bleu(references, candidates,smoothing_function=SmoothingFunction().method4)
    else : 
        candidates= list(row["traductions_fr_en_txt_only"])
        references = [list(row["references_en"])]
        return sentence_bleu(references, candidates,smoothing_function=SmoothingFunction().method4)
    
def save_dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size,epoch = 0):
    df = dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size)
    df["bleu_en_fr"] = df.apply(lambda row: bleu(row,"en"), axis=1)
    df["bleu_fr_en"] = df.apply(lambda row: bleu(row,"fr"), axis=1)
    if image_bool:
        df["bleu_en_fr_txt_only"] = df.apply(lambda row: bleu_txt_only(row,"en"), axis=1)
        df["bleu_fr_en_txt_only"] = df.apply(lambda row: bleu_txt_only(row,"fr"), axis=1)
    df.to_csv(str(epoch)+model_fr.prefix+"_eval.csv")
from nltk.translate.bleu_score import sentence_bleu 
from nltk.translate.meteor_score import meteor_score
from importlib import reload as _reload
import greedy_beam_search
from Pipeline import *
_reload(greedy_beam_search)
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
    # 
    with open("logs.txt",'a') as logs :
        if mode == 'greedy':
            output = give_tokens(greedy_beam_search.CCF_greedy(model_A,model_B,data, features, True),model_B.padding_id,model_B.end_id)[j]
        else : 
            output = greedy_beam_search.CCF_beam_search(model_A, model_B, data, 3, features, True)[j]
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
        logs.write(str(tensor_to_sentence(torch.argmax(model_A(data,True,features),dim = 2)[j].view(-1),inv_map_src)))
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

def donne_random(i,j,val_data_en,val_data_fr,batch_size):
    batched_data_en,batched_data_fr=batchify([val_data_en,val_data_fr],batch_size,True)
    src,features = batched_data_en
    tgt,_ = batched_data_fr
    return src[i],features[i],tgt[i]

def evaluation(mode,val_data_en,val_data_fr,batch_size,model_en,model_fr,inv_map_en,inv_map_fr):
    tokenized_val_en = val_data_en[0]
    i = np.random.randint(len(tokenized_val_en)//batch_size)
    j = np.random.randint(batch_size)
    src,features,tgt = donne_random(i,j,val_data_en,val_data_fr,batch_size)
    features = features.to(device,dtype=torch.float32)
    data = [src,features]
    trad,bleu,meteor = traduit(mode,model_en,model_fr,data, inv_map_en,True,tgt[j],inv_map_fr,j) 
    with open("logs.txt",'a') as logs :
        logs.write("\nPhrase à traduire : \n" + tensor_to_sentence(src[j],inv_map_en)+ "\nPhrase traduite : \n"+ trad)
        logs.close()
    return bleu, meteor


#%%
#%% Data reading

import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/Users/lucas/OneDrive/ENSAE/Cours 2A/Statap'app/Codes_modeles"
file_fr = open(data_path+'/multi30k-dataset/data/task1/tok/train.lc.norm.tok.fr',"r")
file_en = open(data_path+'/multi30k-dataset/data/task1/tok/train.lc.norm.tok.en',"r")

list_en = []
list_fr = []

for line in file_en:
  list_en.append(line)

for line in file_fr: 
    list_fr.append(line)

file_en.close()
file_fr.close()

#%% Frequency of occurrence of letters

letters_en = {}
for sentence in list_en:
    for letter in sentence: 
        if letter in letters_en.keys(): 
            letters_en[letter] += 1
        else: 
            letters_en[letter]=1 

letters_fr = {}
for sentence in list_fr:
    for letter in sentence: 
        if letter in letters_fr.keys(): 
            letters_fr[letter]+=1
        else: 
            letters_fr[letter]=1


def dict_normalization(d):
    factor=1.0/sum(d.values())
    res = {}
    for k in d:
        res[k] = d[k]*factor
    return res

def histogram(list_dict,legend):
    for i in range(len(list_dict)): 
        d = list_dict[i]
        transformed_d = {key: value for key, value in sorted(d.items()) if key.isalpha()}
        plt.bar(transformed_d.keys(), dict_normalization(transformed_d).values(),alpha=1/len(list_dict),label=legend[i])
        plt.legend(loc='upper right') 

## Plotting 

histogram([letters_en, letters_en],["Letters in Frnech","Letters in English"])
# Does it work well ? 

#%% Pairs of letters

pairs_fr={}

for sentence in list_fr:
    for i in range(len(sentence)-1): 
        if sentence[i] in pairs_fr.keys(): 
            if sentence[i+1] in pairs_fr[sentence[i]].keys(): 
                pairs_fr[sentence[i]][sentence[i+1]] += 1
            else: 
                pairs_fr[sentence[i]][sentence[i+1]]=1
        else: 
            pairs_fr[sentence[i]] = {sentence[i+1]:1}


pairs_en = {}

for sentence in list_en:
    for i in range(len(sentence)-1): 
        if sentence[i] in pairs_en.keys(): 
            if sentence[i+1] in pairs_en[sentence[i]].keys(): 
                pairs_en[sentence[i]][sentence[i+1]] += 1
            else: 
                pairs_en[sentence[i]][sentence[i+1]]=1
        else: 
            pairs_en[sentence[i]] = {sentence[i+1]:1}

#%% Matrix of following letter

n_en = len(pairs_en)
n_fr = len(pairs_fr)

followers_en = np.zeros(n_en)
followers_fr = np.zeros(n_fr)



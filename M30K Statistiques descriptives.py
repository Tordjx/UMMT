#%%
import numpy as np
data_path = "C:/Users/valen/Documents/ENSAE/STATAPP"
fichier_FR= open(data_path+'/multi30k-dataset/data/task1/tok/train.lc.norm.tok.fr',"r")
fichier_EN= open(data_path+'/multi30k-dataset/data/task1/tok/train.lc.norm.tok.en',"r")


liste_EN = []
for ligne in fichier_EN:
  liste_EN.append(ligne)
liste_FR = []
for ligne in fichier_FR : 
    liste_FR.append(ligne)
fichier_EN.close()
fichier_FR.close()

#%%Frequence d'appartition des lettres 
 
lettres_fr = {}
for phrase in liste_FR :
    for lettre in phrase : 
        if lettre in lettres_fr.keys() : 
            lettres_fr[lettre]+=1
        else : 
            lettres_fr[lettre]=1
lettres_en = {}
for phrase in liste_EN :
    for lettre in phrase : 
        if lettre in lettres_en.keys() : 
            lettres_en[lettre]+=1
        else : 
            lettres_en[lettre]=1

def normaliser_dico (d):
    factor=1.0/sum(d.values())
    d_ = {}
    for k in d:
        d_[k] = d[k]*factor
    return d_

#%%Traçons des histogrammes
import matplotlib.pyplot as plt 
lettres_fr = {key: value for key, value in sorted(lettres_fr.items())}
lettres_en = {key: value for key, value in sorted(lettres_en.items())}
plt.bar(lettres_fr.keys(), normaliser_dico(lettres_fr).values(),alpha=0.5,label="FR")
plt.bar(lettres_en.keys(), normaliser_dico(lettres_en).values(),alpha = 0.5,label = 'EN')
plt.legend(loc='upper right')

#%%En ne gardant que les lettres
lettres_fr = {k: v for k, v in lettres_fr.items() if k.isalpha()}
lettres_en = {k: v for k, v in lettres_en.items() if k.isalpha()}
plt.bar(lettres_fr.keys(), normaliser_dico(lettres_fr).values(),alpha=0.5,label="FR")
plt.bar(lettres_en.keys(), normaliser_dico(lettres_en).values(),alpha = 0.5,label = 'EN')
plt.legend(loc='upper right')

#%%
import numpy as np
import matplotlib.pyplot as plt 
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

def Tracer_histogramme(dicos,legende) :
    for i in range(len(dicos)) : 
        Dico = dicos[i]
        Transformé = {key: value for key, value in sorted(Dico.items()) if key.isalpha()}
        plt.bar(Transformé.keys(), normaliser_dico(Transformé).values(),alpha=1/len(dicos),label=legende[i])
        plt.legend(loc='upper right')   
#%%Traçons des histogrammes
Tracer_histogramme([lettres_en,lettres_fr],['EN','FR'])

#%%Faisons des paires de lettres
PairesFR={}
for phrase in liste_FR :
    for i in range(len(phrase)-1): 
        if phrase[i] in PairesFR.keys() : 
            if phrase[i+1] in PairesFR[phrase[i]].keys() : 
                PairesFR[phrase[i]][phrase[i+1]]+=1
            else : 
                PairesFR[phrase[i]][phrase[i+1]]=1
        else : 
            PairesFR[phrase[i]]={phrase[i+1]:1}

PairesEN = {}
for phrase in liste_EN :
    for i in range(len(phrase)-1): 
        if phrase[i] in PairesEN.keys() : 
            if phrase[i+1] in PairesEN[phrase[i]].keys() : 
                PairesEN[phrase[i]][phrase[i+1]]+=1
            else : 
                PairesEN[phrase[i]][phrase[i+1]]=1
        else : 
            PairesEN[phrase[i]]={phrase[i+1]:1}

#%%Frequence d'apparition des lettres sachant que la précédente est un t
lettre = 'e'
Tracer_histogramme([PairesEN[lettre],PairesFR[lettre]], ['EN','FR'])
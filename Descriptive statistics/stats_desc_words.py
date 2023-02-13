#%% Data reading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import *
from patsy import dmatrices

# aprés avoir implémenté subword nmt (bpe) faire la frequence des sous mots comme les lettres
# avoir le nombre de mots uniques par langue
# nb mots dans le jeu de données test qui ne sont pas dans entrainement
# Faire des visuels pour le rapport (nuage de mots)

#%% Reading data

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

#%% Values

words_en = [ line.split() for line in list_en ]
words_en = [ item for sublist in words_en for item in sublist if item != '.\n' ]

words_fr = [ line.split() for line in list_fr ]
words_fr = [ item for sublist in words_fr for item in sublist if item != '.\n' ]

sizes_en = pd.Series(words_en).value_counts()

sizes_fr = pd.Series(words_fr).value_counts()

n_en = len(sizes_en)
n_fr = len(sizes_fr)


sizemin_en = sizes_en[-1]
sizemin_fr = sizes_fr[-1]

def f(size, size_min, n):
    return log10( size/(size_min*n) )

values_en = np.array([ f(sizes_en[i], sizemin_en, n_en) for i in range(n_en-1)])
values_fr = np.array([ f(sizes_fr[i], sizemin_fr, n_fr) for i in range(n_fr-1)])

true_labels_en = np.array([ log10(r+1) for r in range(n_en-1) ])
true_labels_fr = np.array([ log10(r+1) for r in range(n_fr-1) ])

#%% Regression en 

df_en = pd.DataFrame()
df_en["values"] = values_en
df_en["label"] = true_labels_en

y_en, X_en = dmatrices('label ~ values', data=df_en, return_type='dataframe')
model_en = sm.OLS(y_en, X_en)
results_en = model_en.fit()
const_en = results_en.params[0]
beta_en = results_en.params[1]
# print(results_en.summary())

predicted_en = [ const_en + beta_en*x for x in values_en ]

plt.figure()
plt.scatter(values_en, predicted_en)
plt.scatter(values_en, true_labels_en, color='r')
plt.plot(values_en, predicted_en)
plt.title("English")
plt.show()

#%% Regression fr 

df_fr = pd.DataFrame()
df_fr["values"] = values_fr
df_fr["label"] = true_labels_fr

y_fr, X_fr = dmatrices('label ~ values', data=df_fr, return_type='dataframe')
model_fr = sm.OLS(y_fr, X_fr)
results_fr = model_fr.fit()
const_fr = results_fr.params[0]
beta_fr = results_fr.params[1]
# print(results_fr.summary())

predicted_fr = [ const_fr + beta_fr*x for x in values_fr ]

plt.figure()
plt.scatter(values_fr, predicted_fr)
plt.scatter(values_fr, true_labels_fr, color='r')
plt.plot(values_fr, predicted_fr)
plt.title("French")
plt.show()


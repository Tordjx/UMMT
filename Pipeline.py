#%%
import numpy as np
import torch
data_path = "C:/Users/valen/Documents/ENSAE/STATAPP/"
train_text_path = data_path+'multi30k-dataset/data/task1/tok/train.lc.norm.tok.'
train_img_path = data_path+"/Images/train-resnet50-res4frelu.npy"
eval_text_path = data_path+'multi30k-dataset/data/task1/tok/val.lc.norm.tok.'
eval_img_path = data_path+"/Images/val-resnet50-res4frelu.npy"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1


class Langue():
    def __init__(self,nom_langue) :
        self.nom_langue = nom_langue
        self.word2index= {}
        self.word2count = {}
        self.n_mots = 0 
        self.index2word = {0: "SOS", 1: "EOS"}
        self.train_fichier_txt =None
        self.train_fichier_img =None
        self.eval_fichier_txt =None
        self.eval_fichier_img =None
        self.train_data=[]
        self.eval_data=[]
        self.ouvrir_data()
        self.data_to_tensor()
    def ouvrir_data(self):
        self.train_fichier_txt = open(train_text_path+self.nom_langue)
        self.eval_fichier_txt = open(eval_text_path+self.nom_langue)
        # self.train_fichier_img = np.load(train_img_path)
        self.eval_fichier_img = np.load(eval_img_path)
        # i = 0
        # for ligne in self.train_fichier_txt : 
        #     self.train_data.append([ligne,self.train_fichier_img[i]])
        #     self.ajouter_ligne(ligne)
        #     i+=1
        i=0
        for ligne in self.eval_fichier_txt:
            self.eval_data.append([ligne,self.eval_fichier_img[i] ])
            self.ajouter_ligne(ligne)
            i+=1
        
    def ajouter_ligne(self,ligne):
        for mot in ligne.split(" "):
            self.ajouter_mot(mot)
    def ajouter_mot (self, mot):
        if mot not in self.word2index:
            self.word2index[mot] = self.n_mots
            self.word2count[mot] = 1
            self.index2word[self.n_mots] = mot
            self.n_mots += 1
        else:
            self.word2count[mot] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def data_to_tensor(self) : 
        # self.train_data = [ [self.tensorFromSentence(data[0]) , data[1]]    for data in self.train_data]
        self.eval_data = [ [self.tensorFromSentence(data[0]) , data[1]]    for data in self.eval_data]

Francais = Langue("fr")
Anglais = Langue('en')
# %%

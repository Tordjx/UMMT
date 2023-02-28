#%% libraries
import torch
import clip
from PIL import Image
import os
from torchvision.datasets import CIFAR100


device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import matplotlib.pyplot as plt



#%% loading data

images = np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")

#%% CLIP model example

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("C:/Users/lucas/Documents/GitHub/UMMT/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


#%% Get text 

def get_vocab() : 
    fichier_vocab_fr = open('C:/Users/lucas/Documents/GitHub/UMMT/vocab.fr')
    fichier_vocab_en = open('C:/Users/lucas/Documents/GitHub/UMMT/vocab.en')
    vocab_en = [line.split()[0] for line in fichier_vocab_en if len(line.split()) == 2]
    vocab_en = dict((y,x) for (x,y) in enumerate(vocab_en))
    vocab_fr = [line.split()[0] for line in fichier_vocab_fr if len(line.split()) == 2]
    vocab_fr = dict((y,x) for (x,y) in enumerate(vocab_fr))
    fichier_vocab_en.close()
    fichier_vocab_fr.close()
    return vocab_en, vocab_fr


def get_train_data():
    vocab_en,vocab_fr = get_vocab()
    fichier_train_fr = open('C:/Users/lucas/Documents/GitHub/UMMT/train.BPE.fr')
    fichier_train_en = open('C:/Users/lucas/Documents/GitHub/UMMT/train.BPE.en')
    train_data_fr = [ligne.strip().split(" ") for ligne in fichier_train_fr ]
    train_data_en = [ligne.strip().split(" ") for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()

    for ligne in train_data_en: 
        for mot in ligne : 
            if mot not in vocab_en: 
                vocab_en[mot] = len(vocab_en.keys())

    for ligne in train_data_fr: 
        for mot in ligne: 
            if mot not in vocab_fr: 
                vocab_fr[mot] = len(vocab_fr.keys())

    embedded_fr = [torch.tensor([vocab_fr[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_fr]
    embedded_en = [torch.tensor([vocab_en[x]  for x in ligne ], dtype= torch.long) for ligne in train_data_en]

    return embedded_en, embedded_fr



#%% Similarity computation 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model 
model, preprocess = clip.load('ViT-B/32', device)

# Data 
# images = np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")
text_en, text_fr = get_train_data()
print(text_en[34].shape)



#%% 

cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
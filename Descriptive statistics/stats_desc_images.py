#%% libraries
import torch
import clip
from PIL import Image
import os
from torchvision.datasets import CIFAR100
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random as rd

device = "cuda" if torch.cuda.is_available() else "cpu"


# Get correspondance between images and captions

def get_captions():
    # file names
    file = open("C:/Users/lucas/Documents/GitHub/dataset/data/task1/image_splits/train.txt")
    names = [line[:-1] for line in file ]
    file.close()
    # Captions
    file = open("C:/Users/lucas/Desktop/train.en")
    captions = [line[:-1] for line in file ]
    file.close()
    return names, captions

names, captions = get_captions()
dict_captions = { names[i] : captions[i] for i in range(len(names)) }

#%% Similarity prediction computation

model, preprocess = clip.load('ViT-B/32', device)

def similarity_prediction(file_name, nb=1):
    # image 
    im = Image.open(r"C:/Users/lucas/Downloads/flickr30k-images.tar/flickr30k-images/"+ file_name) 
    image = preprocess(im).unsqueeze(0).to(device)
    # text 
    texts = clip.tokenize(names[:100]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(nb)
    print("Top predictions:")
    for value, index in zip(values, indices):
        print("Caption n°" + str(index.item()) + " : " + str(value.item()))

    return values, indices

# Tests
c = 0
for i in range(100):
    values, indices = similarity_prediction(names[i])
    for value, index in zip(values, indices):
        if index.item() == i:
            c += 1
print(c/100)


#%% Similarity score for each image-text couple

model, preprocess = clip.load('ViT-B/32', device)

def similarity_score():
    simi_scores = {}

    for file_name, caption in dict_captions.items():
        # image
        im = Image.open(r"C:/Users/lucas/Downloads/flickr30k-images.tar/flickr30k-images/"+ file_name) 
        image = preprocess(im).unsqueeze(0).to(device)
        # text 
        text = clip.tokenize([caption]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)

        simi_scores[file_name] = similarity
    
    return simi_scores

simi_scores = similarity_score()

#%% Comparaison with other values

def similarity_comparaison(file_name,nb=10,printing=False):

    im = Image.open(r"C:/Users/lucas/Downloads/flickr30k-images.tar/flickr30k-images/"+ file_name) 
    image = preprocess(im).unsqueeze(0).to(device) 
    caption = dict_captions[file_name]
    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)

    other_captions = rd.choice(list(captions),size=nb)
    similarities = []
    for cap in other_captions:
        with torch.no_grad():
            cap_features = model.encode_text(cap)
            cap_features /= cap_features.norm(dim=-1, keepdim=True)
            similarities.append((100.0 * image_features @ cap_features.T))

    if printing:
        print("Real similarity : " + str(similarity))
        for i in range(len(similarities)):
            print("Random caption " + str(i) + " : " + str(similarities[i]))
    
    similarities.insert(0,similarity)
    return similarities

# Tests : 
for file_name, caption in dict_captions.items():
    similarity_comparaison(file_name, printing=False)


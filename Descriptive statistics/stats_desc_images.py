#%% libraries
import torch
import clip
from PIL import Image
import os
from torchvision.datasets import CIFAR100
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

#%% Get correspondance between images and captions

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

#%% Similarity computation

model, preprocess = clip.load('ViT-B/32', device)

def similarity(file_name, nb=1):
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

    # logits_per_image, logits_per_text = model(image, texts)
    # probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    # print("Label probs:", probs)

    return values, indices

c = 0
for i in range(100):
    values, indices = similarity(names[i])
    for value, index in zip(values, indices):
        if index.item() == i:
            c += 1
print(c/100)
    


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

def similarity(file_name):
    # image 
    im = Image.open(r"C:/Users/lucas/Downloads/flickr30k-images.tar/flickr30k-images/"+ file_name) 
    image = preprocess(im).unsqueeze(0).to(device)
    # text 
    texts = clip.tokenize(names).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(5)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(value + " " + index)

similarity(names[0])


#%% Test tuto 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)


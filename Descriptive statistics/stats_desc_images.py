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
import csv
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import pylab
from scipy.stats import kstest, norm

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

#%% Show images

im = Image.open(r"C:/Users/lucas/Downloads/flickr30k-images.tar/flickr30k-images/83482568.jpg")
plt.imshow(im)
plt.show()

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

        simi_scores[file_name] = similarity.item()
    
    return simi_scores

simi_scores = similarity_score()

with open('similarity_score.csv', 'w') as f:
    for key in simi_scores.keys():
        f.write("%s,%s\n"%(key,simi_scores[key]))

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

    other_captions = np.random.choice(list(captions),size=nb)
    similarities = []
    for cap in other_captions:
        cap = clip.tokenize([cap]).to(device)
        with torch.no_grad():
            cap_features = model.encode_text(cap)
            cap_features /= cap_features.norm(dim=-1, keepdim=True)
            similarities.append((100.0 * image_features @ cap_features.T).item())

    if printing:
        print("Real similarity : " + str(similarity))
        for i in range(len(similarities)):
            print("Random caption " + str(i) + " : " + str(similarities[i]))
    
    similarities.insert(0,similarity.item())
    return similarities

# Tests : 
results_comparaison = {}
for file_name, caption in dict_captions.items():
    results_comparaison[(file_name,caption)] = similarity_comparaison(file_name, printing=False)

with open('comparaison_score.csv', 'w') as f:
    for key in results_comparaison.keys():
        f.write("%s,%s\n"%(key,results_comparaison[key]))

#%% Distribution for similarity 

simi_scores = pd.read_csv("similarity_score.csv",names = ["file", "score"])
sorted_simi = np.array(list(simi_scores["score"]))
sorted_simi.sort()

mu, std = norm.fit(sorted_simi) 

plt.figure()
plt.hist(sorted_simi, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

# Test for the gaussian distribution 

# qqplot :
sm.qqplot(sorted_simi, line='45')
pylab.show()
# KS test :
ks_statistic, p_value = kstest(sorted_simi, 'norm')
print(ks_statistic, p_value)

#%% Distribution comparaison score

comparaison_score = pd.read_csv("comparaison_score.csv",sep=')',on_bad_lines='skip', names = ["file", "list_values"])

simi_score = []
comp_score = []
for i in range(len(comparaison_score["list_values"])):
    a=comparaison_score["list_values"][i].split(",")
    del a[0]
    a[0] = a[0][1:]
    a[-1] = a[-1][:-1]
    b = [ float(x) for x in a]
    simi_score.append(b[0])
    comp_score.append(b[1:])

#%% Treatment 

# Comparaison simi_score vs mean of other scores
simi_score_index = [ (simi_score[i], i) for i in range(len(simi_score)) ]
simi_score_index.sort()
simi_score, permutation = zip(*simi_score_index)
index = [ i for i in range(len(simi_score)) ]
comp_score = [ comp_score[i] for i in permutation ] # permute the other scores with the sorting permutation

# Mean :
mean_comp_score = [ np.mean(x) for x in comp_score ] 
plt.figure()
plt.plot(index, simi_score, color="red")
plt.scatter(index, comp_score, color="blue", s=0.01 )
plt.show()

# Difference : 
diff_comp_score = [ simi_score[i] - mean_comp_score[i] for i in range(len(simi_score)) ]
plt.figure()
plt.plot(index, simi_score, color="red")
plt.scatter(index, diff_comp_score, s=0.01)
plt.show()

# histrogram 
sorted_comp_score = sorted(mean_comp_score)
mu_c, std_c = norm.fit(sorted_comp_score) 

plt.figure()
plt.hist(sorted_comp_score, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_c = norm.pdf(x, mu_c, std_c)
plt.plot(x, p_c, 'k', linewidth=2)
plt.show()

#%% Both
# Both histogram : 
plt.figure()
plt.hist(sorted_comp_score, density=True, label="means of other scores", color="blue")
plt.hist(sorted_simi, density=True, label="similarity scores",color="red")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_c = norm.pdf(x, mu_c, std_c)
plt.plot(x, p_c, 'k', linewidth=2)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.legend()
plt.show()


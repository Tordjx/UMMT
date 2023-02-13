#%% libraries
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import matplotlib.pyplot as plt



#%% loading data

images = np.load("C:/Users/lucas/Documents/GitHub/UMMT/val-resnet50-res4frelu.npy")

# Test 
image_test = images[0]
# image_test = image_test.reshape((1024,196))

plt.figure()
plt.imshow(image_test, interpolation='nearest')
plt.show()

#%% CLIP model

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("C:/Users/lucas/Documents/GitHub/UMMT/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


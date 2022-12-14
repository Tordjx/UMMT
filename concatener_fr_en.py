#%%
#Train file : concatenation de train fr et train en 
path = "C:/Users/valen/Documents/ENSAE/STATAPP/"
train_fr = path + "multi30k-dataset/data/task1/tok/train.lc.norm.tok.fr"
train_en = path  + "multi30k-dataset/data/task1/tok/train.lc.norm.tok.en"
filenames = [train_fr, train_en]

with open(path + "trainfile_subword", 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


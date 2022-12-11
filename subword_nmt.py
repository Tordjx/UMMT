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

import os
# os.system('cmd /c "./learn_bpe.py -s 10000 -o {codes_file}"')

Query1 = "C:/Users/valen/Documents/ENSAE/STATAPP/multi30k-dataset/scripts/subword-nmt/learn_bpe.py -s 10000 -o C:/Users/valen/Documents/ENSAE/STATAPP/codes_files_subword"

# Query = "C:/Users/valen/Documents/ENSAE/STATAPP/multi30k-dataset/scripts/subword-nmt/learn_bpe.py --input C:/Users/valen/Documents/ENSAE/STATAPP/multi30k-dataset/data/task1/tok/train.lc.norm.tok.fr C:/Users/valen/Documents/ENSAE/STATAPP/multi30k-dataset/data/task1/tok/train.lc.norm.tok.en -s 10000 -o C:/Users/valen/Documents/ENSAE/STATAPP/codes_files_subword --write-vocabulary C:/Users/valen/Documents/ENSAE/STATAPP/Vocab.fr C:/Users/valen/Documents/ENSAE/STATAPP/Vocab.en"

"C:/Users/valen/Documents/ENSAE/STATAPP/multi30k-dataset/scripts/subword-nmt/learn_bpe.py"

os.system('cmd /c "' + Query1 + '"')
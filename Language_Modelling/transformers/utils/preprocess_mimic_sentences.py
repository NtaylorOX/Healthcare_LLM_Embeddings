import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import spacy
from spacy.lang.en import English
import nltk
import re
from tqdm import tqdm
import string

import os
import time
from argparse import ArgumentParser
import itertools
tqdm.pandas()

### setup args
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default="/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/")
parser.add_argument('--save_dir', type=str,
                    default="/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/")
parser.add_argument('--sample', action='store_true', default=False
                    )
parser.add_argument('--sample_size', type=int,
                    default=100,
                    help = "Number of rows to sample from the training and test data")
parser.add_argument('--merge_short_sentences', action='store_true', default=False)
args = parser.parse_args()

start_time = time.time()
##########################



#The pre-processing pipeline inherits from the original ClinicalBERT with minor changes. see https://github.com/kexinhuang12345/clinicalBERT
data_dir = args.data_dir

save_dir = args.save_dir


# train data
train_df = pd.read_csv(f"{data_dir}/lm_pretraining_train_250000.csv")


#test data
test_df = pd.read_csv(f"{data_dir}/lm_pretraining_test_1000.csv")

# ordinarily we use a sample of 250k
# set train and test filenames
train_filename = "all_train_sentences_250000.txt"
test_filename = "all_test_sentences_1000.txt"

#sample 100 for testing
if args.sample:
    print(f"Sampling {args.sample_size} rows from train and test data...")
    train_df = train_df.sample(n=args.sample_size, random_state=42)
    test_df = test_df.sample(n=args.sample_size, random_state=42)
    train_filename = "all_train_sentences_100.txt"
    test_filename = "all_test_sentences_100.txt"

print(f"Train df shape: {train_df.shape}")
print(f"Test df shape: {test_df.shape}")
# load spacy model-
# nlp = English()
# nlp.add_pipe('sentencizer')

# function to convert to spacy sentences and merge any that are too short
def toSentence(x):    
    doc = nlp(x)
    text=[]
    try:
        for sent in doc.sents:
            st=str(sent).strip() 
            if len(st)<30:
                #Merging too-short sentences to appropriate length, this is inherited from ClinicalBERT with changes in merged length 
                if len(text)!=0:
                    text[-1]=' '.join((text[-1],st))
                else:
                    text=[st]
            else:
                text.append((st))
    except:
        print(doc)
    return text


    


# load sci spacy model
nlp = spacy.load('en_core_sci_sm', disable=['tagger','ner'])
nlp.max_length = 20000000

# convert all training notes to list of sentences
# combine all rows of "text" column
# print(f"train df text: {train_df['TEXT']}")
# remove na - maybe overkill at this point but doesn't hurt
train_df = train_df.dropna(subset=['TEXT'])

# convert train df to list of sentences and join together as one huge string
train_text = ' '.join(train_df['TEXT'].tolist())

# same for test data
test_df = test_df.dropna(subset=['TEXT'])
test_text = ' '.join(test_df['TEXT'].tolist())


print(f"Training data has {len(train_text)} total characters and test data has {len(test_text)} total characters")
if args.merge_short_sentences:
    print(f"Will be merging short sentences...")
    
    # using spacy on the whole string will likely run into substantial memory issues
    # all_train_sentences = toSentence(train_text)
    # whilst slower we can isntead pass each row and have it separately processed
    with tqdm(total=len(train_df)) as pbar:       
        train_sentences = train_df["TEXT"].progress_apply(lambda x: toSentence(x))
        all_train_sentences =  [item for sublist in train_sentences for item in sublist]
        pbar.update()
    # 
    # all_train_sentences = toSentence(train_text)    

    print(f"Number of spacy derived sentences for the TRAIN set: {len(all_train_sentences)}")
    with open(f"{save_dir}/{train_filename}", "w") as f:
        for s in all_train_sentences:
            
            f.write(s + "\n")
    f.close()
    print(f"Working on test data...")
    
    # all_test_sentences = toSentence(test_text)
    with tqdm(total=len(test_df)) as pbar:
        
        test_sentences = test_df["TEXT"].progress_apply(lambda x: toSentence(x))
        all_test_sentences =  [item for sublist in test_sentences for item in sublist]
        pbar.update()
    
    
    print(f"Number of spacy derived sentences for the TEST set: {len(all_test_sentences)}")
    with open(f"{save_dir}/{test_filename}", "w") as f:
        for s in all_test_sentences:
            f.write(s + "\n")    
else:
    # first convert to dataframe column to docs
    print(f"Working on train data...")
    docs = list(nlp.pipe(train_df['TEXT'], disable=['tagger','ner'], batch_size=2000, n_process=16))
    
    # now convert to list of sentences
    sentences = [sent for doc in docs for sent in doc.sents]
    
    with open(f"{save_dir}/{train_filename}", "w") as f:
        for s in tqdm(sentences):
            # print(f"s: {s}")
            f.write(s.text + "\n")
    f.close()
    
    print(f"Working on test data...")
    # now test data
    docs = list(nlp.pipe(test_df['TEXT']))
    
    # now convert to list of sentences
    sentences = [sent for doc in docs for sent in doc.sents]
    
    with open(f"{save_dir}/{test_filename}", "w") as f:
        for s in tqdm(sentences):
            # print(f"s: {s}")
            f.write(s.text + "\n")
    f.close()
    
    
    
    
    # # old way - try convert one big string to spacy doc
    # # convert to spacy doc if possible? - not sure if all data will be handled as RAM is limited
    # doc = nlp(train_text)

    # print(f"Number of spacy derived sentences for the TRAIN set: {len(list(doc.sents))}")


    # with open("./all_train_sentences.txt", "w") as f:
    #     # Loop over the sentences, writing each one to the file
    #     for sentence in tqdm(doc.sents):
    #         f.write(sentence.text + "\n")

    # # Close the file
    # f.close()

    # print(f"Working on test data...")
    # # test
    # doc = nlp(test_text)
    # print(f"Number of spacy derived sentences for the TEST set: {len(list(doc.sents))}")
    # with open("./all_test_sentences.txt", "w") as f:
    #     # Loop over the sentences, writing each one to the file
    #     for sentence in tqdm(doc.sents):
    #         f.write(sentence.text + "\n")

    # f.close()

end_time = time.time()
print(f"Time taken: {end_time - start_time} ")
# all_train_sentences = list(doc.sents)

# # save to file
# with open("./all_train_sentences.txt", "w") as f:
#     for s in all_train_sentences:
#         f.write(s + "\n")


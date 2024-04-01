import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from loguru import logger

# from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
# from transformers import RobertaTokenizerFast as RobertaTokenizer
# from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.model_selection import train_test_split


import argparse
from datetime import datetime
import warnings

from dataset_processing import encode_classes, convert_to_binary_classes, FewShotSampler

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

'''
Script to create training datasets for classification tasks with a few shot sampler - will take N samples per class to allow training/testing on any sample size with a balanced dataset


Example cmd usage for creating fewshot mimic note-category pseudo classification with classes reduced to top 8:

python .\data_utils\create_fewshot_data.py --data_dir {data_path} --reduce_classes fewshot --few_shot_n --dataset pseudo_classification

Example for using the mimic icd9-triage task:

python create_fewshot_dataset.py --data_dir /mnt/sdc/niallt/mimic3-icd9-data/intermediary-data/triage/ --few_shot_n 128 --dataset icd9-triage

'''


def main():
    parser = argparse.ArgumentParser()

    #TODO - add an argument to specify whether using balanced data then update directories based on that

    # Required parameters
    parser.add_argument("--data_dir",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/", # for triage /mnt/sdc/niallt/mimic3-icd9-data/intermediary-data/triage/
                        help = "The data path to the directory containing the text data csv file with labels")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The filename of the training file containing the text and labels")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the validation file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of the test file")


    parser.add_argument("--text_col",
                        default = "text",
                        type=str,
                        help = "col name for the column containing the text")

    parser.add_argument("--save_dir",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/",
                        help = "The data path to save the created dataset"
                        )
    
    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")
    parser.add_argument("--binary_class_transform",
                        action = 'store_true',
                        help="Whether not to convert a multi-class problem into a binary one") # this is fairly specific to a dataset we developed
    parser.add_argument("--binary_severity_split_value",
                        default = 3,
                        type=int,
                        help = "The mid point value ranging from 0 - N_classes to split into a binary set of classification i.e. 0/1" # this is fairly specific to a dataset we developed
                        )
    parser.add_argument("--reduce_classes",                     
                        
                        action = "store_true",
                        help = "For datasets with many classes - whether or not we want to reduce this by taking only top K classes" 
                        )

    parser.add_argument("--n_classes_keep",
                        default = 8,
                        type=int,
                        help = "Only valid if reduce_classes True. The number of classes to keep, i.e. keep top K classes." 
                        )

    parser.add_argument(
        "--dataset",
        default="pseudo_classification", #or icd9-triage: 
        type=str,
        help="name of dataset",
    )

    parser.add_argument(
        "--label_col",
        default="label", # label column of dataframes provided - should be label if using the dataprocessors from utils
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=4,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )



    parser.add_argument(
        "--few_shot_n",
        type=int,
        default = None
    )


    args = parser.parse_args()

    print(f"arguments provided are: {args}")



    # for now we are obtaining all data from one training/test file. So we can load in now and do subsetting for actual datasets

    few_shot_n = args.few_shot_n


    # first we may need to do some task specific preprocessing 
    if args.dataset == "pseudo_classification":
        logger.warning(f"Using the following dataset: {args.dataset} ")

        # update data and save_dir based on task
        data_dir = f"{args.data_dir}/"
        
        # here we only want to save a maximum of 10k samples to align better with other datasets
        
        # set text and label column names
        text_col = "TEXT"
        cat_col = "CATEGORY"
        create_test_val_split = True
        # load in the training and test data - will split into test/val later
        train_df = pd.read_csv(f"{data_dir}/train_df_notes_interim_preprocessed.csv")
        test_df = pd.read_csv(f"{data_dir}/test_df_notes_interim_preprocessed.csv")
        
        # pull out the columns we need - text and label
        train_df = train_df[[text_col,cat_col]]
        test_df =  test_df[[text_col,cat_col]]        

        
    elif args.dataset == "icd9-triage":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        
        # update data and save_dir based on task
        data_dir = f"{args.data_dir}"
        # set text and label column names
        text_col = "text"
        cat_col = "triage-category"
        create_test_val_split = False # we already have them for the triage task
        # load in the training and test data - will split into test/val later
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        val_df = pd.read_csv(f"{data_dir}/valid.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        
        
        
        # pull out the columns we need - text and label
        train_df = train_df[[text_col,cat_col]]
        val_df = val_df[[text_col,cat_col]]
        test_df =  test_df[[text_col,cat_col]]                
           
    else:
       raise NotImplementedError 
    
   
    # # if binary_transform - convert labels from range 0-N to 0/1
    # if args.binary_class_transform:
    #     logger.warning(f"Converting to binary classification problem")
    #     #update save dir
        
    #     train_df = convert_to_binary_classes(df = train_df, split_value = args.binary_severity_split_value)
    #     val_df = convert_to_binary_classes(df = val_df, split_value = args.binary_severity_split_value)
    #     test_df = convert_to_binary_classes(df = test_df, split_value = args.binary_severity_split_value)
    
    # now create val split 
    if create_test_val_split:
        test_df, val_df = train_test_split(test_df, test_size=0.5, random_state = 1)
    
    # if we are reducing classes - ONLY FOR NOTE CATEGORY TASK
    if args.reduce_classes:
        if args.dataset == "icd9-triage":
            raise NotImplementedError("Cannot reduce classes for triage task")
        logger.warning(f"Will be reducing the number of classes to keep down to: {args.n_classes_keep} classes!")
        classes_to_keep = list(train_df[cat_col].value_counts().keys()[:args.n_classes_keep])
        # get new DFs with only the top N classes in
        
        train_df = train_df[train_df[cat_col].isin(classes_to_keep)]
        val_df = val_df[val_df[cat_col].isin(classes_to_keep)]
        test_df = test_df[test_df[cat_col].isin(classes_to_keep)]
    # get class label encodings based on training data    
            
    # now encode the labels - and sort by the value counts rather than string value - this well help keep ordering when subetting by class frequency
    class_list, idx_to_class, class_to_idx = encode_classes(train_df,
                                                            label_col=cat_col,
                                                            sort_by_value_count=True)
    
    logger.warning(f"Class labels: {class_list}\n\nidx_to_class:{idx_to_class}\n\nclass_to_idx:{class_to_idx}")

    # convert create label column from categories mapped to label int
    train_df['label'] = train_df[cat_col].map(class_to_idx)
    val_df['label'] = val_df[cat_col].map(class_to_idx)
    test_df['label'] = test_df[cat_col].map(class_to_idx)        


    # initialise the sampler
    if few_shot_n is not None:
        logger.warning(f"Will be using fewshot sampler with :{args.few_shot_n} samples per class")
        support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False, label_col = args.label_col)
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = support_sampler(train_df.to_dict(orient="records"), seed = 1)

        # do we actually want to resample the val and test sets - probably not? 
        val_df = support_sampler(val_df.to_dict(orient="records"), seed = 1)
        test_df = support_sampler(test_df.to_dict(orient="records"), seed = 1)
    
    # set dataset save path
    dataset_save_path = f"{args.save_dir}/{args.dataset}/"


    # now write these to file
    if args.dataset == "icd9-triage":
        if "no_category_in_text" in args.data_dir:
            dataset_save_path = f"{dataset_save_path}/no_category_in_text/"
    
    if args.binary_class_transform:
        dataset_save_path = f"{dataset_save_path}/binary_class/"
        
    if args.reduce_classes:
        
        dataset_save_path = f"{dataset_save_path}/class_reduced_{args.n_classes_keep}/"
        
    if args.few_shot_n is not None:       
        
        dataset_save_path = f"{dataset_save_path}/fewshot_{args.few_shot_n}/"
    

        
    # create if it doesn't exist
    if not os.path.exists(f"{dataset_save_path}"):
        os.makedirs(f"{dataset_save_path}")   
    logger.warning(f"Saving files to: {dataset_save_path}")
    # now write each dataframe to file
    train_df.to_csv(f"{dataset_save_path}/train.csv", index = None)
    val_df.to_csv(f"{dataset_save_path}/valid.csv", index = None)
    test_df.to_csv(f"{dataset_save_path}/test.csv", index = None)    


    
if __name__ == "__main__":
    main()
    




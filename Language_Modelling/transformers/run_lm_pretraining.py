import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# # print(os.en)
# os.environ["CUDA_VISIBLE_DEVICES"]="7,8" #6,7 # 2080s = 0,3,5,6,8
import pandas as pd
import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, Trainer, TrainingArguments, pipeline
from peft import get_peft_config, get_peft_model, LoraConfig, IA3Config,TaskType
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PeftConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    prepare_model_for_int8_training,
    # AutoPeftModel,
    prepare_model_for_kbit_training # only for latest dev version of peft
)
from loguru import logger
import string

import random
import argparse
from datetime import datetime, timedelta
import functools
import yaml
import json

'''
Script to run language modelling training using the HF trainer class.

Provide training and test data paths to your own data

python run_lm_pretraining.py --sample --train_sample_size 250000 --test_sample_size 1000 --max_steps 50000 --train_batch_size 4 --eval_batch_size 2 --mlm

# Example usage for dev run to check all works with only 1k samples etc
python run_lm_pretraining.py --training_text_data_path {training_data_file} --test_text_data_path {test_data_file} --hf_model_name bert-base-uncased --sample --train_sample_size 1000 --test_sample_size 200 --max_steps 200 --save_every_steps 20 --eval_every_steps 20  --mlm



 Example usage for DDP: 
 CUDA_VISIBLE_DEVICES=7,8 torchrun --nproc_per_node 2 run_lm_pretraining.py --sample --train_sample_size 250000 --test_sample_size 1000 --max_steps 100000 --train_batch_size 8 --eval_batch_size 8 --mlm





## Example for mlm
python run_lm_pretraining.py --training_text_data_path {training_data_file} --test_text_data_path {test_data_file} --hf_model_name bert-base-uncased --max_steps 100000 --save_every_steps 500 --eval_every_steps 500  --mlm


#Example usage for causal language modelling
python run_lm_pretraining.py --training_text_data_path {training_data_file} --test_text_data_path {test_data_file} --hf_model_name bert-base-uncased --max_steps 100000 --save_every_steps 500 --eval_every_steps 500

## Models to try:
- bigscience/bloom-350m
- roberta-base or roberta-large
- allenai/biomed_roberta_base

facebooks bio-lms: github.com/facebookresearch/bio-lm 

'''


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--training_text_data_path",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/",
                        help = "The data path to directory containing the formatted language modelling training data")
    parser.add_argument("--test_text_data_path",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/",
                        help = "The data path to directory containing the formatted language modelling training data")

    parser.add_argument("--hf_model_name",
                        default = "roberta-base",
                        type=str,
                        help = "The data path to the file containing the local hf pretrained models or the name of the hf model when connected to internet")                        

    parser.add_argument("--cache_dir",
                        default = "/mnt/sdc/niallt/.cache", 
                        type=str,
                        help = "The directory to save and subsequently load all transformer downloaded models/processed datasets etc.")
                        
    parser.add_argument("--save_path",
                        type=str,
                        default = "/mnt/sdc/niallt/saved_models/language_modelling/mimic/",
                        help = "The directory to save the trained model")
    parser.add_argument("--custom_model_name",
                        default = "mimic",
                        type=str,
                        help = "The custom string to add to the save path to distinguish this model from its base version")   
    parser.add_argument("--logging_dir",
                        default = "/mnt/sdc/niallt/saved_models/language_modelling/mimic/logs/",
                        type=str,
                        help = "The root directory to save the tensorboard logs - the folders will be created dynamically based on model used etc.")                       

    parser.add_argument("--mlm",
                        action = "store_true",
                        help = "Whether or not to run masked language modelling objective")
    parser.add_argument("--max_steps",
                        default = 100000,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--warmup_steps",
                        default = 200,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--eval_every_steps",
                        default = 2000,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--save_every_steps",
                        default = 2000,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--log_every_steps",
                        default = 50,
                        type=int,
                        help = "How often are we logging?")
    parser.add_argument("--block_size",
                        default = 512,
                        type=int,
                        help = "this is ultimately the max tokenized sequence length which will be used to divide the concatenated version of the entire text stream into chunks of block_size")
    parser.add_argument("--train_batch_size",
                        default = 12,
                        type=int,
                        help = "the size of training batches")
    parser.add_argument("--eval_batch_size",
                        default = 12,
                        type=int,
                        help = "the size of evaluation batches")
    parser.add_argument("--max_epochs",
                        default = 30,
                        type=int,
                        help = "the maximum number of epochs to train for")
    parser.add_argument("--grad_accum_steps",
                        default = 1,
                        type=int,
                        help = "the number of update steps to accumulate gradients for, before performing a backward/update pass ")
    parser.add_argument("--learning_rate",
                        default = 2e-5,
                        type=int,
                        help = "the learning rate for the step/weight updates - acts as initial learning rate for AdamW optimizer")  
    parser.add_argument("--weight_decay",
                        default = 0.01,
                        type=float,
                        help = "the weight decay to apply to to all layers except all bias and LayerNorm weights in AdamW optimizer")
    parser.add_argument("--train_sample_size",
                        default = 250000,
                        type=int,
                        help = "The sample size for the training data - this will be used to create the filename to find")
    parser.add_argument("--test_sample_size",
                        default = 1000,
                        type=int,
                        help = "The sample size for the test data - this will be used to create the filename to find") 
    parser.add_argument("--saving_strategy",
                        default = "steps",
                        type=str,
                        help = "The saving strategy to use. For details, see: https://huggingface.co/docs/transformers/main_classes/trainer") 
    parser.add_argument("--evaluation_strategy",
                        default = "steps",
                        type=str,
                        help = "The saving strategy to use. For details, see: https://huggingface.co/docs/transformers/main_classes/trainer")     
    
    parser.add_argument("--text_col",
                        default = "text",
                        type=str,
                        help = "The name of the column with the text data in")
    parser.add_argument("--sample",
                        action = "store_true",
                        help = "Whether or not to process a sub sample of the data - primarily for dev purposes")

    parser.add_argument("--apply_LORA",
                        action = "store_true",
                        help = "Whether to apply LORA to the model for MLM training")
    parser.add_argument("--lora_target_modules",
                        default = None,
                        type = list,
                        help = "The target modules of the model to apply LORA to e.g. key, value, query")
    
    
    args = parser.parse_args()    


    logger.info(f"args provided are: {args}")


    # get datetime now to append to save dirs
    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M")

    # data_paths
    training_text_data_path = f"{args.training_text_data_path}/train.txt"
    test_text_data_path = f"{args.test_text_data_path}/test.txt"
    hf_model_name = args.hf_model_name
    custom_model_name = args.custom_model_name

    
        
    # change save dirs if LORA is applied
    if args.apply_LORA:
        save_path = f"{args.save_path}/{hf_model_name}-{custom_model_name}-wecho-LORA/{time_now}/"
    else:
        save_path = f"{args.save_path}/{hf_model_name}-{custom_model_name}-wecho/{time_now}/"
    # change save dirs if sampled
    if args.sample:
        logger.info(f"Getting a subsample of the data!")
        training_text_data_path = f"{args.training_text_data_path}/with_echo_train_{args.train_sample_size}.txt"
        test_text_data_path = f"{args.test_text_data_path}/with_echo_test_{args.test_sample_size}.txt"
        # change save dirs if LORA is applied
        if args.apply_LORA:
            save_path = f"{args.save_path}/{hf_model_name}-{custom_model_name}-wecho-LORA/sampled_{args.train_sample_size}/{time_now}/"
        else:
            
            save_path = f"{args.save_path}/{hf_model_name}-{custom_model_name}-wecho/sampled_{args.train_sample_size}/{time_now}/"
    
    


    # update logging dir with save path
    logging_dir = f"{save_path}/logs/"
    
    logger.warning(f"Saving model to: {save_path}!!!!!!!!!!")
 

    logger.info(f"The model to be finetuned is: {hf_model_name}")    

    # load tokenizer and model

    tokenizer = AutoTokenizer.from_pretrained(f"{hf_model_name}", cache_dir = args.cache_dir)

    # some tokenizers may not have a "pad" token, which can throw errors with the collator functions if padding is required
    if not tokenizer.pad_token:
        logger.warning(f"Tokenizer had no pad token set - will be setting to eos token id i.e. {tokenizer.eos_token_id}")
        tokenizer.pad_token = tokenizer.eos_token
    # if we are doing masked language modelling
    if args.mlm:
        logger.warning(f"Will be performing Masked Language Modelling")
        model = AutoModelForMaskedLM.from_pretrained(f"{hf_model_name}", cache_dir = args.cache_dir)
    else:        
        logger.warning(f"Will be performing Causal or AutoRegressive Language Modelling")
        model = AutoModelForCausalLM.from_pretrained(f"{hf_model_name}", cache_dir = args.cache_dir)
        
        
    ##################### setup LORA #####################
    def unfreeze_model(model):
        for param in model.parameters():
            param.requires_grad = True
    
    # using LORA?
    if args.apply_LORA:
        logger.info("##################### Applying LORA to the model #####################")
        peft_type = PeftType.LORA
        lr = 3e-4
        peft_config = LoraConfig(task_type=None, inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        # print trainable params
        model.print_trainable_parameters()
        #NOTE - when using the peft_config with no task_type - it will freeze all non-adapter layers including the classifier

    # if the tokenizer has a model_max_length - assert that block_size is not greater than this
    if hasattr(tokenizer, 'model_max_length'):
        assert args.block_size <= tokenizer.model_max_length, "It seems you have tried setting a block size larger than the loaded tokenizers model max length. Try set lower"

    # set up dataset class    

    def tokenize_function(examples):
        '''
        Function to return a tokenized version of the input text

        args:
            examples: datasets object obtained via load_datasets. 

        returns:
            dictionary of tokenized inputs with appropriate input_ids, attention_mask etc.
        '''
        return tokenizer(examples["text"])
        
    def group_texts(tokenized_examples, block_size = 512):
        '''
        Function to concatenate all texts together then split the result into smaller chunks of a specified block_size

        args:
            examples: tokenized dataset produced by the tokenizer_function
            block_size: int -> the chunk or block_size to divide the full concatenated text into
        '''
        examples = tokenized_examples.copy()
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # can use the following line to cut off tails
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0,total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # for both causal and masked language modelling the "right shift" of input text is done by the model internally. Thus for now, labels=input_ids
        result['labels'] = result['input_ids'].copy()
        
        return result
    

    # load dataset into a HF dataset class
    logger.info(f"Loading training data from : {training_text_data_path} and test data from: {test_text_data_path}")
    dataset = load_dataset("text", data_files={"train": f"{training_text_data_path}", "validation": f"{test_text_data_path}"})
    logger.info(f"Creating tokenized datasets!")
    tokenized_datasets = dataset.map(tokenize_function, batched = True, remove_columns = ['text'])    
    # need to use functools partial to be able to supply a block size dynamically
    logger.info(f"Creating lm datasets!")
    lm_datasets  = tokenized_datasets.map(functools.partial(group_texts, block_size = args.block_size), batched = True, batch_size = 1000) # we keep this as 1000 by default, likely won't change
    logger.info(f"The number of text chunks to train on are: {lm_datasets['train'].num_rows}! The number to test on is: {lm_datasets['validation'].num_rows} ")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=args.mlm, mlm_probability=0.15)
    
    
    logger.warning(f"lm_datasets['train'].column_names: {lm_datasets['train'].column_names}")

    # define validation function and metric
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):        
            logits = logits[0]
        return logits.argmax(dim = -1)

    def compute_mlm_accuracy(eval_preds):

        '''
        Function to compute a basic accuracy for MLM - i.e. did the model correctly predict the token at the [MASK] position
        '''
        preds, labels = eval_preds
        
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = (labels != -100)
        labels = labels[mask]
        preds = preds[mask]
        
        # check two ways this accuracy is calculated
        accuracy_manual = (preds == labels).mean()     
               
        return {'accuracy':accuracy_manual}


    def compute_clm_accuracy(eval_preds):

        '''
        Function to compute a basic accuracy for causal/autoregressive LM - i.e. did the model correctly predict the next token. I think generally we just care about the last token prediction?
            or do we want a mean of all...
        '''
        preds, labels = eval_preds

        # print(f"shape of shifted labels: {labels[:,1:].shape} and shifted preds: {preds[:,:-1].shape}")

        # gpt2 based models seem to require shifting post model outputs for loss - so same holds here
        labels = labels[:,1:].reshape(-1)
        preds = preds[:,:-1].reshape(-1)
        
        accuracy_manual = (preds == labels).mean()

        return {'accuracy':accuracy_manual}

    # set up the metric based on whether it is masked language modelling or not
    if args.mlm:
        compute_metrics = compute_mlm_accuracy
    else:
        compute_metrics = compute_clm_accuracy
        
        
    # write the argparser object to file to ensure we can see which hparams were provided
    
    # ensure the save path folder has been created
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f'{save_path}/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    with open(f'{save_path}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
        
    
    # manually save tokenizer to same path as the model
    # tokenizer.save_pretrained(save_path)

    # set up training arguments
    training_args = TrainingArguments(
        output_dir=f"{save_path}/",
        max_steps= args.max_steps,
        num_train_epochs=args.max_epochs,
        label_names = ["labels"], # only use in conjunction with remove_unused_columns = False
        
        per_device_train_batch_size=args.train_batch_size, # seems auto handeled by HF trainer now
        per_device_eval_batch_size = args.eval_batch_size,

        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_every_steps,
               
        save_steps=args.save_every_steps,
        save_total_limit=2,
        load_best_model_at_end = True,
        
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_steps=args.log_every_steps,
        logging_first_step=True,
        logging_strategy = 'steps',
        logging_dir = f"{logging_dir}/",
        remove_unused_columns = False,
        fp16 = True

    )

    # set up the trainer
    trainer = Trainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_datasets["train"],
        eval_dataset = lm_datasets['validation'],
             
        
        # compute_metrics = compute_metrics,
        # preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )
    # run trainer
    trainer.train()
    
    
    # save just the lora weights
    if args.apply_LORA:
        model.save_pretrained(f"{save_path}/adapter_weights/")

if __name__ == "__main__":
    main()


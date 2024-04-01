
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4" #6,7import pandas as pd 2080s = 0,3,5,6,8 Nvidia-smi ids: 0, 3, 5, 6, 8 Actual id: 5,6,7,8,9 

import numpy as np
from datasets import load_dataset, load_metric
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, Trainer, TrainingArguments, pipeline
# from loguru import logger
import string
from transformers.utils import logging
import random
import argparse
from datetime import datetime, timedelta
import functools
import yaml
import json
import sys 

# add the sys path for utils
# sys.path.append("./")
from models.utils.custom_hf_trainer import CustomHFTrainer
from models.mlm_contrastive_transformer import TransformerForPreTraining

from loguru import logger as loguru_logger

# PEFT
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
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


'''
Script to run language modelling training using a custom HF Model and trainer class.

Provide training and test data paths to your own data

Example usagE: 
python run_combined_pretraining.py --train_batch_size 4 --eval_batch_size 2 --compute_contrastive_loss --max_steps 100000

# data parallel
CUDA_VISIBLE_DEVICES=8,9  torchrun --nproc_per_node 2 run_combined_pretraining.py --train_batch_size 8 --eval_batch_size 8 --max_epochs 1 

# note loss only
CUDA_VISIBLE_DEVICES=0 python run_combined_pretraining.py --train_batch_size 8 --eval_batch_size 8 --max_epochs 2 --contrastive_loss_weight 0.6 --compute_note_loss_only --dev_run


# with local model - add this to the end of the command

--hf_model_name /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000

# with LORA
CUDA_VISIBLE_DEVICES=2,4  torchrun --nproc_per_node 2 run_combined_pretraining.py --train_batch_size 8 --eval_batch_size 8 --max_epochs 2 --apply_LORA --contrastive_loss_weight 0.6

'''


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--training_text_data_path",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/lm_pretraining_train_250000.csv",
                        help = "The data path to directory containing the formatted language modelling training data")
    parser.add_argument("--test_text_data_path",
                        type=str,
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/lm_pretraining_test_1000.csv",
                        help = "The data path to directory containing the formatted language modelling training data")

    parser.add_argument("--hf_model_name",
                        default = "roberta-base",
                        type=str,
                        help = "The data path to the file containing the local hf pretrained models or the name of the hf model when connected to internet")                        

    parser.add_argument("--cache_dir",
                        default = None, 
                        type=str,
                        help = "The directory to save and subsequently load all transformer downloaded models/processed datasets etc.")
                        
    parser.add_argument("--save_path",
                        type=str,
                        default = "/mnt/sdc/niallt/saved_models/language_modelling/mimic/",
                        help = "The directory to save the trained model")
    parser.add_argument("--custom_model_name",
                        default = "mimic-note",
                        type=str,
                        help = "The custom string to add to the save path to distinguish this model from its base version")   
    parser.add_argument("--logging_dir",
                        default = "/mnt/sdc/niallt/saved_models/language_modelling/mimic/logs/combined_pretraining/",
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
    parser.add_argument("--contrastive_loss_weight",
                        default = 0.1,
                        type=float,
                        help = "the weight to apply to contrastive loss when combined with MLM loss")
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
    parser.add_argument("--dev_run",
                        action = "store_true",
                        help = "Whether or not to run a dev run and save to testing dir")
    parser.add_argument("--use_auto_model",
                        action = "store_true",
                        help = "Whether or not to use the Transformers AutoClass instead of custom one")
    parser.add_argument("--compute_contrastive_loss",
                        action = "store_true",
                        help = "Whether or not to add contrastive loss to pre-training objective")
    parser.add_argument("--compute_note_loss_only",
                        action = "store_true",
                        help = "Whether or not to only compute note category loss and not mlm loss")
    parser.add_argument("--compute_mlm_loss_only",
                        action = "store_true",
                        help = "Whether or not to only compute mlm loss and not note category loss")
    parser.add_argument("--apply_LORA",
                        action = "store_true",
                        help  = "Whether or not to apply LORA to the model")

    args = parser.parse_args()    

    # get datetime now to append to save dirs
    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M")

    # data_paths

    hf_model_name = args.hf_model_name
    custom_model_name = args.custom_model_name
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - forward slashes really matter for the naming convention make sure to append the path with a forward slash
    if "saved_models" in hf_model_name:
        if "declutr" in hf_model_name:
            if "few_epoch" in hf_model_name:
                if "span_comparison" in hf_model_name:
                    save_model_name = hf_model_name.split("/")[9] + "/declutr/" + hf_model_name.split("/")[-3]
                else:
                    save_model_name = hf_model_name.split("/")[8] + "/declutr/" + hf_model_name.split("/")[-3]

            else:
                save_model_name = hf_model_name.split("/")[7] + "/declutr/" + hf_model_name.split("/")[-3]
        elif "contrastive" in hf_model_name or "custom_pretraining" in hf_model_name:
            save_model_name = hf_model_name.split("/")[7]
        else:
            save_model_name = hf_model_name.split("/")[7]
    else:    
        save_model_name = hf_model_name
        
    

    # are we doing dev run
    if args.dev_run:
        
        if args.apply_LORA:
            save_path = f"/mnt/sdc/niallt/saved_models/code_testing/note_mlm_lora/"
            logging_dir = f"{save_path}/logs/"
        elif args.compute_note_loss_only:
            save_path = f"/mnt/sdc/niallt/saved_models/code_testing/note_mlm_note_loss_only/"
        elif args.compute_mlm_loss_only:
            save_path = f"/mnt/sdc/niallt/saved_models/code_testing/note_mlm_mlm_loss_only/"
        else:
            save_path = f"/mnt/sdc/niallt/saved_models/code_testing/note_mlm/"
            
        logging_dir = f"{save_path}/logs/"

        
        
    elif args.use_auto_model:
        save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-auto_mlm_only/sampled_250000/{time_now}/"
        logging_dir = f"{save_path}/logs/"
    #TODO - edit this logic based on whether MLM only or both objectives
    elif args.compute_contrastive_loss:
        loguru_logger.warning(f"Will be computing contrastive loss!")
        
        # but do we want to just see mlm loss
        if args.compute_mlm_loss_only:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_contrastive_max_epoch_{args.max_epochs}_mlm_loss_only/sampled_250000/{time_now}/"
        
        # are we just computing note loss
        if args.compute_note_loss_only:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_contrastive_max_epoch_{args.max_epochs}_note_loss_only/sampled_250000/{time_now}/"
        
        if args.apply_LORA:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_contrastive_max_epoch_{args.max_epochs}_weighted_lora/sampled_250000/{time_now}/"
        else:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_contrastive_max_epoch_{args.max_epochs}_weighted/sampled_250000/{time_now}/"
        logging_dir = f"{save_path}/logs/" 
    else:
        loguru_logger.warning(f"Will use classification head for the note category loss!")
        # are we just computing mlm
        if args.compute_mlm_loss_only:
            print("will be computing mlm loss only")
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_max_epoch_{args.max_epochs}_mlm_loss_only/sampled_250000/{time_now}/"
            
        # just note loss only?
        elif args.compute_note_loss_only:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_max_epoch_{args.max_epochs}_note_loss_only/sampled_250000/{time_now}/"
        
        else: 
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_max_epoch_{args.max_epochs}_weighted/sampled_250000/{time_now}/"
            
        if args.apply_LORA:
            save_path = f"{args.save_path}/{save_model_name}-{custom_model_name}-custom_pretraining_max_epoch_{args.max_epochs}_weighted_lora/sampled_250000/{time_now}/"
        
            
        logging_dir = f"{save_path}/logs/" 
               
    if args.dev_run:
        dataset = load_dataset("csv", 
                    data_files = {"train":"/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/fewshot_16/train.csv",
                                    "valid":"/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/fewshot_16/valid.csv"},
                    cache_dir = args.cache_dir)     
    else:
        # load in the data     
        dataset = load_dataset("csv", 
                                data_files = {"train":f"{args.training_text_data_path}",
                                                "valid":f"{args.test_text_data_path}"},
                                cache_dir = args.cache_dir)
    # dataset = load_dataset("csv", 
    #                         data_files = {"train":"/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/fewshot_200/train.csv",
    #                                         "valid":"/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/fewshot_200/valid.csv"},
    #                         cache_dir = "/mnt/sdc/niallt/.cache/")
    
    dataset = dataset.rename_column('label', 'category_label')
    
    # get the number of unique labels to define the shape of the classification head or contrastive loss function
    num_pretraining_labels = len(set(dataset['train']['CATEGORY']))
    
    loguru_logger.warning(f"We have {num_pretraining_labels} unique labels for the pre-training objective")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    # model = BertForPreTraining.from_pretrained('bert-base-uncased')
    
    if args.use_auto_model:
        loguru_logger.info(f"Using AutoModel Class")
        model = AutoModelForMaskedLM.from_pretrained(f"{hf_model_name}", cache_dir = args.cache_dir)
    else:
        loguru_logger.info(f"Using new pre-training model")
        # if args.compute_contrastive_loss:
        #     loguru_logger.info("will be training with contrastive loss and MLM!")
        #     model = TransformerForPreTraining.from_pretrained(hf_model_name,
        #                                                       cache_dir = args.cache_dir,
        #                                                       compute_contrastive = True,
        #                                                       contrastive_loss_weight = args.contrastive_loss_weight,
        #                                                       num_pretraining_labels = num_pretraining_labels)
        # else:
        #     loguru_logger.info("will be training with a classification head and MLM!")
        #     model = TransformerForPreTraining.from_pretrained(hf_model_name, cache_dir = args.cache_dir,
        #                                                       compute_contrastive = False,
        #                                                       contrastive_loss_weight = args.contrastive_loss_weight,
        #                                                       num_pretraining_labels = num_pretraining_labels)
        
        loguru_logger.info(f"Using new pre-training model with contrastive loss: {args.compute_contrastive_loss} and LORA: {args.apply_LORA}")
        model = TransformerForPreTraining.from_pretrained(hf_model_name,
                                                            cache_dir = args.cache_dir,
                                                            compute_contrastive = args.compute_contrastive_loss,
                                                            contrastive_loss_weight = args.contrastive_loss_weight,
                                                            compute_note_loss_only = args.compute_note_loss_only,
                                                            compute_mlm_loss_only = args.compute_mlm_loss_only,
                                                            num_pretraining_labels = num_pretraining_labels)
    
    #TODO - refactor                                                          
    def unfreeze_model(model):
        for param in model.parameters():
            param.requires_grad = True
    
    # using LORA?
    if args.apply_LORA:
        loguru_logger.info("##################### Applying LORA to the model #####################")
        peft_type = PeftType.LORA
        lr = 3e-4
        peft_config = LoraConfig(task_type=None,
                                 inference_mode=False,
                                 r=64,
                                 lora_alpha=16,
                                 lora_dropout=0.1,
                                 modules_to_save=["seq_classifier"])
        model = get_peft_model(model, peft_config)
        # print trainable params
        model.print_trainable_parameters()
        #NOTE - when using the peft_config with no task_type - it will freeze all non-adapter layers including the classifier
        # thus we need to unfreeze the classifier layer
        # need to unfreeze the classifier

        # unfreeze_model(model.base_model.seq_classifier)
        # # now after unfreezing
        # loguru_logger.info("##################### After unfreezing classifier #####################")
        # model.print_trainable_parameters()

    
    
    def tokenize_function(examples):
        '''
        Function to return a tokenized version of the input text

        args:
            examples: datasets object obtained via load_datasets. 

        returns:
            dictionary of tokenized inputs with appropriate input_ids, attention_mask etc.
        '''
        return tokenizer(examples["TEXT"], truncation=True, padding = True)

    def preprocess_function(examples):
        
        examples['labels'] = examples['input_ids'].copy()
        return examples
        

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
        
    def preprocess_logits_for_metrics(logits, labels):
        '''
        Function to process the raw logits and labels returned in the evaluation/prediction loop of the HF trainer. With out custom model pre-training
        the returned logits and labels can be a tuple containing logits and labels for both individual pre-training objectives.
        
         
        '''
        if isinstance(logits, tuple):
            if len(logits)==1:        
                logits = logits[0]

                return logits.argmax(dim = -1)
            
            else:
                mlm_logits = logits[0]
                mlm_preds = mlm_logits.argmax(dim= -1)
                cls_logits = logits[1]
                cls_preds = cls_logits.argmax(dim=-1)
                
                return tuple([mlm_preds, cls_preds])
           
    def compute_metrics(eval_preds):

        '''
        Function to compute a basic cls metrics for the pre-training models in this repo. Essentially we want to just compute the metrics for the sequence 
        classification task. 
        '''
        precision_score = load_metric("precision")
        recall_score = load_metric("recall")
        accuracy_score = load_metric("accuracy")
        f1_score = load_metric("f1")   
        
        #TODO - edit this to handle the tuple of logits returned
        preds, labels = eval_preds
        # loguru_logger.info(f"inside compute metric, labels are: {labels} \n\n and preds are: {preds}")
        
        #TODO - check if tuple and what length of each - then if 2 presume it is mlm labels for first and cls for second
        if isinstance(preds, tuple):
            if len(preds) == 1:
                mlm_preds = preds[0]
                mlm_labels = labels[0]
            else:
                mlm_preds = preds[0]
                mlm_labels = labels[0]
                cls_preds = preds[1]
                cls_labels = labels[1]
        
        # calculate cls_metrics
        loguru_logger.info(f"mlm labels:{mlm_labels} \n\n mlm_preds: {mlm_preds}\n\n")
        loguru_logger.info(f"cls labels: {cls_labels} \n\n cls preds: {cls_preds}")
               
        precision = precision_score.compute(predictions=cls_preds, references=cls_labels, average = "macro")["precision"]
        recall = recall_score.compute(predictions=cls_preds, references=cls_labels, average = "macro")["recall"]
        accuracy = accuracy_score.compute(predictions=cls_preds, references=cls_labels)["accuracy"]
        f1_macro = f1_score.compute(predictions=cls_preds, references=cls_labels, average = "macro")["f1"]
        f1_weighted = f1_score.compute(predictions=cls_preds, references=cls_labels, average = "weighted")["f1"]

        
        return {"precision": precision, 
                "recall": recall,
                "accuracy": accuracy,
                "f1_macro":f1_macro,
                "f1_weighted":f1_weighted
                } 
    
                
    # write the argparser object to file to ensure we can see which hparams were provided
    
    # ensure the save path folder has been created
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f'{save_path}/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    with open(f'{save_path}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
        
    
    text_key = "TEXT"

    encoded_dataset = dataset.map(tokenize_function, batched=True, remove_columns = ['TEXT', 'CATEGORY'])
    
    lm_datasets  = encoded_dataset.map(preprocess_function, batched = True, batch_size = 1000)
    
    data_collator = DataCollatorForLanguageModeling(
                                            tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    
    # manually save tokenizer to same path as the model
    # tokenizer.save_pretrained(save_path)
    # set up training arguments
    training_args = TrainingArguments(
        output_dir=f"{save_path}/",
        # max_steps= args.max_steps,
        num_train_epochs=args.max_epochs,
        label_names = ["labels","category_label"],
        
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
        remove_unused_columns=False, # this shouldnt need to be False - but when using PEFT something changes

    )
    
    
    loguru_logger.warning(f"saving logs etc to: {logging_dir}")
    
    
    if args.use_auto_model:
        loguru_logger.info(f"Using AUTO TRAINER")
    # set up the trainer
        trainer = Trainer(
            model=model,
            tokenizer = tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=lm_datasets["train"],
            eval_dataset = lm_datasets['valid'],
            # compute_metrics = compute_metrics,
            # preprocess_logits_for_metrics = preprocess_logits_for_metrics
        )
    # run trainer

    else:
        if args.compute_contrastive_loss:
            loguru_logger.info("################## Using Custom HF Trainer with contrastive loss ##################### \n ")
            # set up the trainer
            trainer = CustomHFTrainer(
                model=model,
                tokenizer = tokenizer,
                args=training_args,
                data_collator=data_collator,
                train_dataset=lm_datasets["train"],
                eval_dataset = lm_datasets['valid']
            )
        else:
            loguru_logger.info("################## Using Custom HF Trainer with classification head ##################### \n ")
            trainer = CustomHFTrainer(
                    model=model,
                    tokenizer = tokenizer,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=lm_datasets["train"],
                    eval_dataset = lm_datasets['valid'],
                    compute_metrics = compute_metrics,
                    preprocess_logits_for_metrics = preprocess_logits_for_metrics
                )           
    # run trainer
    trainer.train()
    
    # # set up training arguments
    # training_args = TrainingArguments(
    #     output_dir=f"{save_path}/",
        # max_steps= args.max_steps,
    #     num_train_epochs=args.max_epochs,
        
    #     # per_device_train_batch_size=args.train_batch_size, # seems auto handeled by HF trainer now
    #     # per_device_eval_batch_size = args.eval_batch_size,

    #     learning_rate = args.learning_rate,
    #     weight_decay = args.weight_decay,
        
    #     evaluation_strategy = args.evaluation_strategy,
    #     eval_steps = args.eval_every_steps,
               
    #     save_steps=args.save_every_steps,
    #     save_total_limit=2,
    #     load_best_model_at_end = True,
        
    #     warmup_steps=args.warmup_steps,
    #     gradient_accumulation_steps=args.grad_accum_steps,
    #     logging_steps=args.log_every_steps,
    #     logging_first_step=True,
    #     logging_strategy = 'steps',
    #     logging_dir = f"{logging_dir}/"

    # )

    # save just the lora weights
    if args.apply_LORA:
        model.save_pretrained(f"{save_path}/adapter_weights/")

    

if __name__ == "__main__":
    main()


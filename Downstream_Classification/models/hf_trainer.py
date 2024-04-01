import os
# set visible cuda device
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #6,7

import torch
import transformers
import torch.nn as nn
from scipy.special import softmax
# import intel_extension_for_pytorch as ipex # for CPU only optimisations



from datasets import load_dataset, load_metric, concatenate_datasets # list_datasets, load_from_disk, DatasetDict, Dataset, load_dataset_builder
import evaluate # this weirdly loads something onto the GPU and will cause OOM on python3.9
import pandas as pd
from tqdm import tqdm
import numpy as np
# custom roberta classificaiton model
# load custom roberta sequence classifier that uses the same averaging as declutr etc
from model_utils import unfreeze_encoder, freeze_n_layers, freeze_encoder, count_trainable_model_parameters
from model_utils import MeanRobertaForSequenceClassification
from model_utils import MeanBertForSequenceClassification
from model_utils import (create_long_model,
                                               RobertaLongForMaskedLM,
                                               RobertaLongForSequenceClassification,
                                               MeanRobertaLongForSequenceClassification)
import sys
sys.path.append("../..")
from data_utils.dataset_processing import FewShotSampler_dataset

import datasets
import random
import pandas as pd

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer
from transformers import LongformerForMaskedLM, LongformerForSequenceClassification
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
import numpy as np

import argparse
import yaml
import json
from loguru import logger as loguru_logger



'''
Script to train classification tasks using the HuggingFace trainer class with custom transformer models.

Example usage:



Triage task
python hf_trainer.py --encoder_model /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/sts_trf_roberta/2_anch_2_pos_min_1024/transformer_format/ --max_epochs 3 \ 
                                            --gpu_idx 7 --freeze_plm --training_data_dir /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/ --task icd9-triage


'''


      

def main():
    parser = argparse.ArgumentParser()

    #TODO - add an argument to specify whether using balanced data then update directories based on that

    # Required parameters
    parser.add_argument("--training_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--eval_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/pseudo_classification/class_reduced_8/",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--cache_dir",
                        default = "/mnt/sdc/niallt/.cache/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the training file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of the test file")

    parser.add_argument("--pretrained_models_dir",
                        default="",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")


    parser.add_argument("--text_col",
                        default = "text",
                        type=str,
                        help = "col name for the column containing the text")

    parser.add_argument("--log_save_dir",
                        default = "/mnt/sdc/niallt/saved_models/pseudo_classification_tasks/mimic/logs/transformers/",
                        type=str,
                        help = "The data path to save tb log files to"
                        )
    parser.add_argument("--ckpt_save_dir",
                    default = "/mnt/sdc/niallt/saved_models/pseudo_classification_tasks/mimic/ckpts/transformers/",
                    type=str,
                    help = "The data path to save trained ckpts to"
                    )

    parser.add_argument("--reinit_n_layers",
                        default = 0,
                        type=int,
                        help = "number of pretrained final bert encoder layers to reinitialize for stabilisation"
                        )
    parser.add_argument("--max_tokens",
                        default = 512,
                        type=int,
                        help = "Max tokens to be used in modelling"
                        )
    parser.add_argument("--max_steps",
                        default = 100000,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--warmup_steps",
                        default = 100,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--eval_every_steps",
                        default = 100,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--save_every_steps",
                        default = 100,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--log_every_steps",
                        default = 10,
                        type=int,
                        help = "How often are we logging?")
    parser.add_argument("--block_size",
                        default = 512,
                        type=int,
                        help = "this is ultimately the max tokenized sequence length which will be used to divide the concatenated version of the entire text stream into chunks of block_size")
    parser.add_argument("--train_batch_size",
                        default = 32,
                        type=int,
                        help = "the size of training batches")
    parser.add_argument("--eval_batch_size",
                        default = 32,
                        type=int,
                        help = "the size of evaluation batches")
    parser.add_argument("--max_epochs",
                        default = 30,
                        type=int,
                        help = "the maximum number of epochs to train for")
    parser.add_argument("--accumulate_grad_batches",
                        default = 1,
                        type=int,
                        help = "number of batches to accumlate before optimization step"
                        )
    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")
    parser.add_argument("--binary_class_transform",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")
    parser.add_argument("--binary_severity_split_value",
                        default = 3,
                        type=int,
                        help = "The severity value ranging from 0 - N severity to split by"
                        )
    parser.add_argument("--class_weights",
                        action = 'store_true',
                        help="Whether not to apply ce_class_weights for cross entropy loss function")

    parser.add_argument("--gpu_idx", 
                        type=int,
                        default=6,
                        help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    parser.add_argument(
            "--encoder_model",
            default= "/mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/",# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased', #emilyalsentzer/Bio_ClinicalBERT'
            type=str,
            help="Encoder model to be used.",
        )
    parser.add_argument(
        "--convert_to_longformer",
        action = "store_true",
        help = "Whether to convert the model to a longformer model",
    )
    
    parser.add_argument(
        "--max_tokens_longformer",
        default=4096,
        type=int,
        help="Max tokens to be considered per instance..",
    )

    parser.add_argument(
        "--encoder_learning_rate",
        default=1e-05,
        type=float,
        help="Encoder specific learning rate.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        default=1e-05,
        type=float,
        help="Classification head learning rate.",
    )
    parser.add_argument(
        "--classifier_hidden_dim",
        default=768,
        type=int,
        help="Size of hidden layer in bert classification head.",
    )

    parser.add_argument(
        "--nr_frozen_epochs",
        default=0,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )
    
    parser.add_argument(
        "--nr_frozen_layers",
        default=-1,
        type=int,         # essentially depends on the transformer model - 0 = do not freeze any of the encoder layers but freezet the embedding layer only, -1 = freeze everything, N > 0 = freeze first N encoder layers 
        help="Number of encoder layers to freeze - only valid when freeze_plm true",
    )

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout value for classifier head.",
    )

    parser.add_argument(
        "--task",
        default="mimic-note-category", # icd9_triage
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--evaluation_strategy",
        default="epoch", # steps or epoch
        type=str,
        help="Whether to log every n steps or per epoch",
    )
    parser.add_argument(
        "--saving_strategy",
        default="no", # steps or epoch or no
        type=str,
        help="Whether to save checkpoints and if so how often",
    )      
    parser.add_argument(
        "--model_type",
        default="mean_embedder", 
        choices = ["automodelforsequence","mean_embedder","longformer"],
        type=str,
        help="This will alter the architecture and forward pass used by transformer sequence classifier. Autosequence will use default class from Transformers library, custom will use our own with adjustments to forward pass",
    )

    parser.add_argument(
        "--label_col",
        default="label", # label column of dataframes provided - should be label if using the dataprocessors from utils
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=24,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="monitor_balanced_accuracy", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=4,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )
    parser.add_argument(
        '--embeddings_analysis',
        default=False,
        type=bool,
        help='Whether to run the embeddings analysis script and save logs etc to make easier to run the analysis'
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor"
    )

    parser.add_argument(
        "--training_size",
        default="full",
        type=str,
        help="full training used, fewshot, or zero"
    )   
    

    parser.add_argument(
        "--few_shot_n",
        type=int,
        default = None
    )
    parser.add_argument(
        "--eval_few_shot_n",
        type=int,
        default = 128
    )
    
    parser.add_argument(
        '--combined_val_test_sets',
        default=False,
        type=bool,
        help='Whether or not to combine the validation and test datasets'
    )
    parser.add_argument(
        '--sensitivity',
        default=False,
        type=bool,
        help='Run sensitivity trials - investigating the influence of number of transformer layers.'
    )

    parser.add_argument(
        '--optimized_run',
        default=False,
        type=bool,
        help='Run the optimized frozen model after hp search '
    )
    parser.add_argument(
        '--freeze_plm',
        action = "store_true",        
        help='Whether to freeze the PLM during fineutuning, i.e. only tune the classification head '
    )
    parser.add_argument(
        '--no_cuda',
        action = "store_true",        
        help='Whether to use cuda/gpu or just use CPU '
    )
    parser.add_argument(
        '--use_ipex',
        action = "store_true",        
        help='Whether to use the ipex optimisation for CPU only.' # see https://huggingface.co/docs/transformers/perf_train_cpu
    )
    parser.add_argument(
        '--fp16_flag',
        action = "store_true",        
        help='Whether to train with fp16.' # see https://huggingface.co/docs/transformers/perf_train_cpu
    )
    parser.add_argument(
        '--hp_search',
        action = "store_true",        
        help='Will this be a run for hyperparameter search?' 
    )
    parser.add_argument('--task_to_keys',
        default = {
                "cola": ("sentence", None),
                "mnli": ("premise", "hypothesis"),
                "mnli-mm": ("premise", "hypothesis"),
                "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"),
                "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "wnli": ("sentence1", "sentence2"),
                "mimic-note-category": ("TEXT", None),
                "icd9-triage":("text", None),
                "icd9-triage-no-category-in-text":("text", None),
                },
        type = dict,
        help = "mapping of task name to tuple of the note formats"
    )

    # TODO - add an argument to specify whether using balanced data then update directories based on that
    args = parser.parse_args()

    loguru_logger.info(f"arguments provided are: {args}")
    # set up parameters
    training_data_dir = args.training_data_dir
    eval_data_dir = args.eval_data_dir
    log_save_dir = args.log_save_dir #NOT USED AT MO
    ckpt_save_dir = args.ckpt_save_dir
    pretrained_dir = args.pretrained_models_dir
    encoder_model = args.encoder_model
    cache_dir = args.cache_dir
    max_tokens = args.max_tokens
    n_epochs = args.max_epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    reinit_n_layers = args.reinit_n_layers
    accumulate_grad_batches = args.accumulate_grad_batches
    model_type = args.model_type
    task = args.task
    few_shot_n = args.few_shot_n
    
    
    #### OLD LOGIC DUMB
    # # are we loading a few_shot trainign set - i.e. a dataset made by our few_shot_sampler with N samples per class
    # if few_shot_n is None:
    #     few_shot_n = "all"    
    # else:
    #     # this logic is based on the presumption that the "../../utils/create_fewshot_dataset.py" script has been used to create datasets
    #     training_data_dir = f"{training_data_dir}/fewshot_{few_shot_n}/"
    # # change training_data_dirs based on the few_shot_n provided
    
    # we have a super aggressively cleaned version of triage dataset with all category text mentions removed as an ablation study
    if task == "icd9-triage":
        if "no_category_in_text" in training_data_dir:
            loguru_logger.warning(f"Running experiment with no category mentions in text!!")
            task = "icd9-triage-no-category-in-text"
        
    
    
    loguru_logger.info(f"Few shot arg provided is: {few_shot_n}")

    # save the hyperparams and arguments to a config file
    # ensure the save path folder has been created

    # set up some checkpoint/ save params

    # TODO clean this up/improve
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - forward slashes really matter for the naming convention make sure to append the path with a forward slash
    if "saved_models" in encoder_model:
        if "declutr" in encoder_model:
            if "few_epoch" in encoder_model:
                if "span_comparison" in encoder_model:
                    model_name = encoder_model.split("/")[9] + "/declutr/" + encoder_model.split("/")[-3]
                else:
                    model_name = encoder_model.split("/")[8] + "/declutr/" + encoder_model.split("/")[-3]

            else:
                model_name = encoder_model.split("/")[7] + "/declutr/" + encoder_model.split("/")[-3]
        elif "contrastive" in encoder_model or "custom_pretraining" in encoder_model:
            model_name = encoder_model.split("/")[7]
        else:
            model_name = encoder_model.split("/")[7] + "/mlm_only/"
    else:    
        model_name = encoder_model.split("/")[-1]
        
    # change name based on longformer
    if args.convert_to_longformer:
        model_name = f"{model_name}_longformer"

    loguru_logger.warning(f"model derived name is: {model_name}")
    # change the save dirs based on task etc
    # span comparison experiments for the declutr models??
    if "span_comparison" in encoder_model:
        loguru_logger.warning(f"Model is being used to compare different span lengths for the declutr algorithm")
        log_save_dir = f"{log_save_dir}/span_comparison/"
        ckpt_save_dir = f"{ckpt_save_dir}/span_comparison/"
    if args.hp_search:
        loguru_logger.warning(f"Running hyperparameter search")
        log_save_dir = f"{log_save_dir}/hp_search/"
        ckpt_save_dir = f"{ckpt_save_dir}/hp_search/"
        # change save strategy to no - do not want to save model per run
        # args.saving_strategy = "no" 
    if args.embeddings_analysis:
        loguru_logger.warning(f"Running experiments for embedding analysis")
        log_save_dir = f"{log_save_dir}/embedding_analysis/"
        ckpt_save_dir = f"{ckpt_save_dir}/embedding_analysis/"
    if args.freeze_plm:
        loguru_logger.warning(f"Freezing part of the model - based on the value of N layers: {args.nr_frozen_layers} ")
        if args.nr_frozen_layers == -1:
            logging_dir = f"{log_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/frozen_plm/"
            ckpt_dir = f"{ckpt_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/frozen_plm/"
        else:
            logging_dir = f"{log_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/frozen_plm_layers_{args.nr_frozen_layers}/"
            ckpt_dir = f"{ckpt_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/frozen_plm_layers_{args.nr_frozen_layers}/"
       
    else:
        logging_dir = f"{log_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/finetuned_plm/"
        ckpt_dir = f"{ckpt_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/finetuned_plm/"   
    
        
    if args.saving_strategy != "no":
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)   

    
    loguru_logger.warning(f"Saving all logs to: {logging_dir}")
    #NOTE we have a separate evaluation data_dir to make a cleaner separation of the training dataset which has undergone fewshot downsampling
    # load the dataset  - if you want test and val separate - but at moment we just combine these for evaluation
    dataset = load_dataset("csv", 
                            data_files = {"train":f"{training_data_dir}/train.csv",
                                            "valid":f"{eval_data_dir}/valid.csv",
                                            "test":f"{eval_data_dir}/test.csv"},
                            cache_dir = None)
    
    # rename label to labels?
    dataset = dataset.rename_column("label", "labels")
    
    num_labels = len(np.unique(dataset['train']['labels']))
    
    
    
    # do few shot sampling of training
    if few_shot_n is not None:
        loguru_logger.info(f"Sampling {few_shot_n} samples per class")
        train_datasets = []
        for label in range(num_labels):
            label_dataset = dataset['train'].filter(lambda x: x['labels'] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(few_shot_n))
            
            # add to list of datasets
            train_datasets.append(label_dataset)

        dataset["train"] = concatenate_datasets(train_datasets)
    
    loguru_logger.info(f"Number of training samples: {len(dataset['train'])}\n and validation samples:{len(dataset['valid'])}")
    
    # can combine datasets by providing dirs as a list
    # dataset = load_dataset("csv", 
    #                         data_files = {"train":f"{training_data_dir}/train.csv",
    #                                         "valid":[f"{training_data_dir}/valid.csv", f"{training_data_dir}/test.csv"],
    #                                         },
    #                         cache_dir = "/mnt/sdc/niallt/.cache/")   

    # set the setnece/task keys
    sentence1_key, sentence2_key = args.task_to_keys[task]
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    # load tokenizer
    loguru_logger.warning(f"Loading tokenizer from model_name_or_path: {encoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_model, cache_dir = cache_dir, use_fast=True, model_max_length = args.max_tokens)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding = "max_length")
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="max_length", return_tensors="pt")

    
    
   
    # here we either use the default automodelforsequenceclassification from Transformers - however this will then change how the embeddings are actually calculated as the "sentence|document" representation - either CLS or pooler etc.
    if model_type == "mean_embedder":
        #NOTE here we check which config the model has - its a bit crude but the models we use likely have either BertConfig or RobertaConfig and require different classification classes
        
        try:
            config = AutoConfig.from_pretrained(encoder_model, cache_dir = cache_dir)
            config_class_name = config.__class__.__name__
        except:
            loguru_logger.warning(f"Couldn't load config - default to regex")
            if "roberta" in encoder_model:
                config_class_name = "RobertaConfig"
            else:
                raise notImplementedError()
                # exit the script
                sys.exit()
        assert config_class_name == "BertConfig" or config_class_name == "RobertaConfig", "Presently the mean_embedder only works for models with Roberta or Bert configs"
        loguru_logger.warning(f"Config Class name is: {config_class_name}")
        loguru_logger.info(f"Using custom mean embedder!")
        if config_class_name == "BertConfig":
            loguru_logger.info(f"We have a model with BERT config: {config}")
            model = MeanBertForSequenceClassification.from_pretrained(encoder_model,
                                                            num_labels=num_labels,
                                                            output_hidden_states = False,
                                                            cache_dir = cache_dir)
        elif config_class_name == "RobertaConfig":
            # loguru_logger.info(f"We have a model with RoBERTa config: {config}")
            
            # are we converting to long?
            if args.convert_to_longformer:
                loguru_logger.info(f"Converting to longformer model")
                # first need to convert both model and tokenizer
                # long model name
                long_model_name = f"/mnt/sdc/niallt/saved_models/longformers/{encoder_model}_longformer"
                #TODO - fix this mess - shouldn't have to save then reload
                long_roberta, tokenizer = create_long_model(encoder_model, encoder_model, long_model_name, 512, 4096)
                
                #NOTE - THIS WILL FUCK UP - see: https://stackoverflow.com/questions/72503309/save-a-bert-model-with-custom-forward-function-and-heads-on-hugginface
                model = MeanRobertaLongForSequenceClassification.from_pretrained(
                                                                            long_model_name,
                                                                            num_labels=num_labels,
                                                                               output_hidden_states=False,
                                                                                 output_attentions=False)
            if "lora" in encoder_model or "LORA" in encoder_model:
                loguru_logger.warning(f"encoder model is: {encoder_model}")
                loguru_logger.warning(f"Found lora weights - will merge into respective model class!")
                # handle models with lora weights
                # load config
                config = PeftConfig.from_pretrained(encoder_model)
                # load base model with the class of interest 
                original_model = MeanRobertaForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                                      num_labels = num_labels,
                                                                                      output_hidden_states = False)
                # load peft model with the lora matrices etc
                reloaded_peft_model = PeftModel.from_pretrained(original_model, encoder_model)
                
                # now merge the weights - not sure this will be foolproof, but seems to be viable if the underlying base model is same
                model = reloaded_peft_model.merge_and_unload()
                
                # the peft library results in all params being frozen - need to unfreeze
                unfreeze_encoder(model)
                
            
            
            else:
                model = MeanRobertaForSequenceClassification.from_pretrained(encoder_model,
                                                                        num_labels=num_labels,
                                                                        output_hidden_states = False,
                                                                        cache_dir = cache_dir)
            

                
        else:
            raise NotImplementedError()
    elif model_type =="automodelforsequence":
        loguru_logger.info("using default automodelforsequence")
        model = AutoModelForSequenceClassification.from_pretrained(encoder_model,
                                                                    num_labels=num_labels,
                                                                    output_hidden_states = False,
                                                                    cache_dir = cache_dir)
            # need to handle gpt2 model and other autoregressive models not having pad token
        if model_name in ["gpt2","opt"]:
            tokenizer.pad_token = tokenizer.eos_token
            loguru_logger.warning(f"got model that needs pad token added, and config is:{model.config}")
            model.config.pad_token_id = model.config.eos_token_id
            
    elif model_type == "longformer":
        loguru_logger.info("using longformer original")
        # has to be "allenai/longformer-base-4096" or clinical longformeer models
        model = LongformerForSequenceClassification.from_pretrained(encoder_model,
                                                                    num_labels=num_labels,
                                                                    output_hidden_states = False
                                                                    )


    # print(f"The sequence classifier model is: {model}")
    
    # freeze the PLM during finetuning or not
    if args.freeze_plm:
        # for now the freeze layers only works for bert and roberta - need to implement for gpt etc
        if model.base_model.config_class.model_type in ["roberta","bert"]:
            print(f"Freezing {args.nr_frozen_layers} layers of the encoder:\n")
            freeze_n_layers(model, args.nr_frozen_layers)
        elif model.base_model.config_class.model_type in ["gpt2", "distilgpt2", "opt"]:
            freeze_encoder(model)
        loguru_logger.warning(f"Trainable params after freezing: {count_trainable_model_parameters(model)}")
    else:
        # we do this just to double make sure all params are unfrozen
        unfreeze_encoder(model)
        
        loguru_logger.warning(f"Trainable params after unfreezing: {count_trainable_model_parameters(model)}")

    # create encoded dataset
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    

    
    print(f"encoded_dataset is: {encoded_dataset}")
    
    # print the length of the input ids for one batch of encoded dataset
    print(f"Length of input ids for one batch of encoded dataset is: {len(encoded_dataset['train']['input_ids'][0])}")

    
    
    #NOTE set output_hidden_states = False - it seems if this is True it will throw an error for the evaluate call of the Trainer.
    # model = AutoModelForSequenceClassification.from_pretrained(encoder_model,
    #                                                            num_labels=num_labels,
    #                                                            output_hidden_states = False,
    #                                                            cache_dir = cache_dir)

    # or we use our custom mean sequence classifiers - altered to take the mean of the last transformer layers hidden representation for all tokens
    # similar to what sentence embeddings does: see https://github.com/JohnGiorgi/DeCLUTR 

    # set up trainer arguments

    metric_name = "f1_macro"


    train_args = TrainingArguments(
        output_dir = f"{ckpt_dir}/",
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_every_steps,
        logging_steps = args.log_every_steps,
        logging_first_step = True,    
        save_strategy = args.saving_strategy,
        save_steps = args.save_every_steps,
        learning_rate=2e-5,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        logging_dir = f"{logging_dir}/",
        save_total_limit=5,
        report_to = 'tensorboard',
        fp16  = args.fp16_flag,
        overwrite_output_dir=True, # will avoid building up lots of files
        no_cuda = args.no_cuda, # for cpu only
        use_ipex = args.use_ipex, # for cpu only
        gradient_accumulation_steps = accumulate_grad_batches,
    )

    # for multiple metrics return a dictionary
    def compute_metrics(eval_pred):
        
        # can't remember why this is being loaded here - rather than once outside? 
        precision_score = load_metric("precision")
        recall_score = load_metric("recall")
        accuracy_score = load_metric("accuracy")
        f1_score = load_metric("f1")        
        roc_auc_score = load_metric("roc_auc", "multiclass")        

        logits, labels = eval_pred        

        
        # print(f"logits are: {logits} of shape: {logits.shape}")
        #TODO add softmax to convert logits to probs
        # print(f"logits shape is: {logits.shape}")
        pred_scores = softmax(logits, axis = -1)        
        predictions = np.argmax(logits, axis = -1)
        
        print(f"Labels are: {labels}\n")
        print(f"Preds are: {predictions}")
        precision = precision_score.compute(predictions=predictions, references=labels, average = "macro")["precision"]
        recall = recall_score.compute(predictions=predictions, references=labels, average = "macro")["recall"]
        accuracy = accuracy_score.compute(predictions=predictions, references=labels)["accuracy"]
        f1_macro = f1_score.compute(predictions=predictions, references=labels, average = "macro")["f1"]
        f1_weighted = f1_score.compute(predictions=predictions, references=labels, average = "weighted")["f1"]
        # roc_auc has slightly different format - needs the probs/scores rather than predicted labels
        roc_auc = roc_auc_score.compute(references=labels,
                                        prediction_scores = pred_scores,
                                        multi_class = 'ovr', 
                                        average = "macro")['roc_auc']
        
        return {"precision": precision, 
                "recall": recall,
                "accuracy": accuracy,
                "f1_macro":f1_macro,
                "f1_weighted":f1_weighted,
                "roc_auc_macro":roc_auc}
        
    def hp_compute_metrics(eval_pred):
            # for hp search just prioritize f1_macro
            f1_score = load_metric("f1")      

            logits, labels = eval_pred
            
            # print(f"logits are: {logits} of shape: {logits.shape}")
            #TODO add softmax to convert logits to probs
            # print(f"logits shape is: {logits.shape}")
            pred_scores = softmax(logits, axis = -1)        
            predictions = np.argmax(logits, axis = -1)

            f1_macro = f1_score.compute(predictions=predictions, references=labels, average = "macro")["f1"]

            
            return {
                    "f1_macro":f1_macro
                    }


    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "valid"
    
    # setup normal trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
        data_collator=collate_fn,

        )
    
    loguru_logger.warning(f"Logs will be saved to: {logging_dir}")
    #### HP search ####
    if args.hp_search:
        loguru_logger.warning(f"Running hyperparameter search!!!")
        def model_init():
            model = AutoModelForSequenceClassification.from_pretrained(encoder_model, num_labels=num_labels, cache_dir=cache_dir)
            if model_name in ["gpt2","opt"]:
                tokenizer.pad_token = tokenizer.eos_token
                loguru_logger.warning(f"got model that needs pad token added, and config is:{model.config}")
                model.config.pad_token_id = model.config.eos_token_id
                
            # also need to freeze layers if specified
                if args.freeze_plm:
                # for now the freeze layers only works for bert and roberta - need to implement for gpt etc
                    if model.base_model.config_class.model_type in ["roberta-base","bert"]:
                        print(f"Freezing {args.nr_frozen_layers} layers of the encoder:\n")
                        freeze_n_layers(model, args.nr_frozen_layers)
                    elif model.base_model.config_class.model_type in ["gpt2", "distilgpt2"]:
                        freeze_encoder(model)
                    print(f"Trainable params: {count_trainable_model_parameters(model)}")
                else:
                    # we do this just to double make sure all params are unfrozen
                    unfreeze_encoder(model)
                    
                    print(f"Trainable params: {count_trainable_model_parameters(model)}")
            return model
        
        def my_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3,10),
                "seed": trial.suggest_int("seed", 1, 40),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4,8,16,32,64]),
            }
            
        hp_trainer = Trainer(            
            args = train_args,            
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset[validation_key],
            tokenizer=tokenizer,
            model_init = model_init,
            compute_metrics=hp_compute_metrics # just using hp_compute_metrics to prioritize f1_macro          
            
            )
        # run the search
        best_run = hp_trainer.hyperparameter_search(direction="maximize", n_trials=50, backend="optuna", hp_space=my_hp_space)        
        
        
        # set log and ckpt dir to reflect best params
        logging_dir = f"{logging_dir}/best_run/"
        ckpt_dir = f"{ckpt_dir}/best_run/"  
        
        # make the dirs if not exist
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)  
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)          
        loguru_logger.warning(f"Best run is: {best_run}, will train new model and save: {logging_dir} and {ckpt_dir}")        
  
        # now use best runs args to train the model
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        
        # also update logging and ckpt dir
        setattr(trainer.args, "logging_dir", logging_dir)
        setattr(trainer.args, "output_dir", ckpt_dir)

        # save the args/params to a text/yaml file
        with open(f'{logging_dir}/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        with open(f'{logging_dir}/config.yaml', 'w') as f:
            yaml.dump(args.__dict__, f)            
           
        with open(f'{logging_dir}/best_run_params.yaml', 'w') as f:
            yaml.dump(trainer.args.__dict__, f)    
        
            
        # re-run with new args
        trainer.train()
        
    
    
    else:

        # run training
        trainer.train()
        
        # run evaluation on test set
        # trainer.evaluate()
        # save the args/params to a text/yaml file
        with open(f'{logging_dir}/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        with open(f'{logging_dir}/config.yaml', 'w') as f:
            yaml.dump(args.__dict__, f) 
        # also save trainer args
        with open(f'{logging_dir}/all_trainer_args.yaml', 'w') as f:
            yaml.dump(trainer.args.__dict__, f)       

# run script
if __name__ == "__main__":
    main()
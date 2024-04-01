# NHS LM Embedding spaces

Repo containing code for the following paper:
"Developing Healthcare Language Model Embedding Spaces" - https://arxiv.org/abs/2403.19802

Repo is a work in progress and will be updated with more details and instructions.
# Overview
We explore three pre-training approaches, all initialised from open Pre-trained Language Models (PLMs). The majority of this work utilises the roberta-base model as the initial model. The reason is we are seeking good embeddings for clinical documents with an emphasis on what is achievable for **resource-limited** environments, such as the NHS. Furthermore, sentence or document embedding models that are achieving SoTA or near SoTA are often reliant on these smaller PLMs e.g. SentenceTransformers etc. 
## Datasets used
We focus on clinical datasets or EHRs. As we are operating in a resource-contrained environment, we opt to not train with the full available datasets and sub-sample 250 thousand clinical documents for each respective dataset. The scaling to full datasets is beyond the scope of this work.

### Mimic-III ...

Accessible with data agreement from Physionet.org...
### NHS Patient Safety Incident Reports
NHS Patient Safety Incident reports from 2020/2021

### Oxford Health Foundation Trust - NHS 

Oxford Health Foundation Trust - NHS clinical notes from 2021.

Note: The NHS datasets are not publicly available.

# Instructions

## Pre-processing
### Mimic-III

Currently inside a [notebook](./Language_Modelling/transformers/utils/preprocess_mimic.ipynb). This will be moved to a script in the future. 


## Language Model Pre-training



### Continued pre-training - MLM

### DeCLUTR pre-training

### Note Category contrastive loss pre-training

This is similar to the original BERT pre-training where the standard MLM objective is combined with a classification objective. The difference is that we are using the note category as the classification objective. This is a multi-class classification task. 

The loss function used is either straight forward cross-entropy loss with a classification head on top of the PLM or a contrastive loss which directly acts on the embeddings produced by the PLM. The contrastive loss is a modified version of the NT-Xent loss. The difference is that we are using the note category as the positive example and a random note category as the negative example. The random note category is sampled from the same batch.

#### Pre-training with cross-entropy loss via classification head

With the LM training datasets curated above, we can now pre-train the PLM with the classification head. The classification head is a simple linear layer with a softmax activation. The loss function is the standard cross-entropy loss. Note you need to provide data as a csv that has both the text and corresponding note category. The text should be in a column called 'text' and the note category should be in a column called 'note_category'. 

An example of running the pre-training with the classification head, run the following command:

```python
# cd ./Language_Modelling/transformers/models
python run_combined_pretraining.py --train_batch_size 4 --eval_batch_size 2 --compute_contrastive_loss --max_steps 100000 --training_text_data_path {your_training_data_path} --test_text_data_path {your_test_data_path}
```

An example of running the pre-training with the contrastive loss function, run the following command:

```python
# cd ./Language_Modelling/transformers/models
python run_combined_pretraining.py --train_batch_size 4 --eval_batch_size 2 --max_steps 100000 --training_text_data_path {your_training_data_path} --test_text_data_path {your_test_data_path}
```

## Downstream task training and evaluation

### Classification
#### Few-shot classification
For most of the tasks we can also look at using different sized training sets. This is useful for the NHS as we can see how well the models perform with limited training data.

The few shot sampler is found [here]('./pseudo_classification_tasks/utils/dataset_processing.py'). Whilst not ideal, we create the fewshot_n datasets and store them separately before running training. 

##### Mimic-III Tasks
An example of creating fewshot_n datasets for the mimic-iii dataset is [here](./pseudo_classification_tasks/utils/create_fewshot_dataset.py) .'

Example usage:

```python
python create_fewshot_dataset.py --data_dir /mnt/sdc/niallt/mimic3-icd9-data/intermediary-data/triage/ --few_shot_n 128 --dataset icd9-triage
```


## Embedding exploration
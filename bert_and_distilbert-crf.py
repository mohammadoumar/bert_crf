#!/usr/bin/env python
# coding: utf-8

# # BERT and DistilBERT for NER (Conll2003 dataset)

# - In the no-finetune case, we freeze the weights of BERT instead of finetuning them.
# - This is equivalent to using BERT embedding for NER.
# - **IMPORTANT** IN THIS CASE, THE LEARNING RATE HAS TO BE INCREASED FROM 1E-5 TO 1E-3 FOR THE CLASSIFIER TO BE ABLE TO LEARN (otherwise this is not working)!


# ========== #
# Librairies #
# ========== #


# pytorch version '1.9.0a0+df837d0'

# !pip install pandas==1.3.4
# !pip install transformers==4.12.5
# !pip install datasets==1.15.1
# !pip install seqeval

# Seems like the tbe dataset Conll2003 is gugging in the latest versions of datasets
# !pip install datasets==1.7.0 


import os
import pickle

import argparse

from collections import Counter

# import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy as np
import torch.nn as nn
import torch

#import transformers
# from transformers import BertTokenizer
#from transformers import BertModel, BertPreTrainedModel
#from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import DataCollatorForTokenClassification
# from transformers.models.bert.modeling_bert import *

from models import BERT_CRF, DistilBERT_CRF

import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets import Dataset
from datasets import ClassLabel
from datasets import load_metric



# ========= #
# Arguments #
# ========= #


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--finetune_mode", type=int, default=1)

parser.add_argument("--results_folder", type=str, default='/raid/home/jeremiec/Data/NER/BERT_and_DistilBERT-CRF/')

parser.add_argument("--nb_epochs", type=float, default=5.)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--eval_steps", type=int, default=100)

args = parser.parse_args()



# ================ #
# Global variables #
# ================ #


RESULTS_FOLDER = args.results_folder

MODEL_NAME = args.model+'-base-cased'

FINETUNE_MODE = args.finetune_mode

RESULTS_FILE = f'{args.model}-crf_finetune-{FINETUNE_MODE}.results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('RESULTS_FILE:', RESULTS_FILE)



# ======= #
# Dataset #
# ======= #


dataset = load_dataset('conll2003', cache_dir='..')

label_names = [list(np.array(x)) for x in dataset['train']['ner_tags']]
label_names = set([item for sublist in label_names for item in sublist])
num_labels = len(label_names)
label_names, num_labels



# ========= #
# Tokenizer #
# ========= #


# https://huggingface.co/docs/transformers/master/en/custom_datasets#tok_ner

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # distilbert-base-uncased

# With this modification, we "only label the first token of a given word and 
# assign -100 to the other subtokens from the same word", as written here.
# https://huggingface.co/docs/transformers/master/en/custom_datasets#tok_ner
# Their code does not do this correctly

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], 
                                 truncation=True, 
                                 padding=True, 
                                 max_length=512, 
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(label[word_idx])
                previous_word_idx = word_idx # *** modif ***
            # *** modif ***
            elif word_idx == previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(-100)
                previous_word_idx = word_idx
            # *** end modif ***

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset_tok = dataset.map(tokenize_and_align_labels, batched=True)



# ===== #
# Model #
# ===== #


NB_EPOCHS = args.nb_epochs
BATCH_SIZE = args.batch_size

if FINETUNE_MODE:
    LR = 1e-5
else:
    LR = 1e-3

data_collator = DataCollatorForTokenClassification(tokenizer)


if args.model == 'bert':
    model = BERT_CRF.from_pretrained(MODEL_NAME, num_labels=num_labels)
else:
    model = DistilBERT_CRF.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(device)


if not FINETUNE_MODE:

    # Freeze weights of the 'bert' part of the model
    if args.model == 'bert':
        for param in model.bert.parameters():
            param.requires_grad = False
    elif args.model == 'distilbert':
        for param in model.distilbert.parameters():
            param.requires_grad = False
        
    # Unfreeze weights of the 'classifier' part of the model (just ot be sure)
    for param in model.classifier.parameters():
        param.requires_grad = True


training_args = TrainingArguments(
    # output
    output_dir=RESULTS_FOLDER,          
    
    # params
    num_train_epochs=NB_EPOCHS,               # nb of epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # cf. paper Sun et al.
    learning_rate=LR,                         # (cf. paper Sun et al.)
#     warmup_steps=500,                       # number of warmup steps for learning rate scheduler
    warmup_ratio=0.1,                         # cf. paper Sun et al.
    weight_decay=0.01,                        # strength of weight decay
    # eval
    evaluation_strategy="steps",              # cf. paper Sun et al.
    eval_steps=args.eval_steps,               # cf. paper Sun et al.
    # log
    logging_dir=RESULTS_FOLDER+f'logs-{args.model}_finetune_{FINETUNE_MODE}',  
    logging_strategy='steps',
    logging_steps=args.eval_steps,
    # save
    save_strategy='steps',
    save_total_limit=2,
    # save_steps=20, # default 500
    load_best_model_at_end=True,              # cf. paper Sun et al.
    metric_for_best_model='eval_loss'
    # metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset_tok['train'],
    eval_dataset=dataset_tok['validation'],
    data_collator=data_collator,
    # compute_metrics=compute_metrics
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

results = trainer.train()

training_time = results.metrics["train_runtime"]

# save best model
# trainer.save_model(os.path.join(RESULTS_FOLDER, f'checkpoint-best-{args.model}_finetune_{FINETUNE_MODE}'))



# ======= #
# Results #
# ======= #


metric = load_metric("seqeval")

# # load model
# model_file = os.path.join(RESULTS_FOLDER, f'checkpoint-best-{args.model}_finetune_{FINETUNE_MODE}')
# model = AutoModelForTokenClassification.from_pretrained(model_file, num_labels=num_labels)
# model.to(device)

model.eval()

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

test_trainer = Trainer(model, data_collator=DataCollatorWithPadding(tokenizer))
test_preds, test_labels, _ = test_trainer.predict(dataset_tok['test'])
# test_preds = np.argmax(test_raw_preds, axis=2)


true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(test_preds, test_labels)
]

true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(test_preds, test_labels)
]

test_results = metric.compute(predictions=true_predictions, references=true_labels)

results = (test_results, training_time)

with open(os.path.join(RESULTS_FOLDER, RESULTS_FILE), 'wb') as fh:
    pickle.dump(results, fh)


del tokenizer
del model
del trainer
del test_trainer
torch.cuda.empty_cache()
import pandas as pd
import json
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import os


#### Check device

print(torch.has_mps)
device = torch.device('mps')

#### Set paths

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = "data"
full_path = os.path.join(WORKING_DIR, DATA_DIR)

#### Load my dataset

train = pd.read_csv(os.path.join(full_path,"train_coarse.csv"),sep='\t')
test = pd.read_csv(os.path.join(full_path,"test_coarse.csv"), sep="\t")
# train = pd.read_csv(os.path.join(full_path,"train_fine.csv"),sep='\t')
# test = pd.read_csv(os.path.join(full_path,"test_fine.csv"), sep="\t")


#### Convert to Huggingface dataset

train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

#### Get labels

labels = [i for i in train['labels'].values.tolist()]
unique_labels = set()

for lb in labels:
    if lb not in unique_labels:
        unique_labels.add(lb)
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

#### Load tokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

def preprocess_function(examples):
    text = examples["text"]
    int_labels = [labels_to_ids[lb] for lb in examples["labels"]]
    examples["labels"] = int_labels
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

tokenized_data_train = train_dataset.map(preprocess_function, batched=True, remove_columns="text")
tokenized_data_test = test_dataset.map(preprocess_function, batched=True, remove_columns="text")

#### Get 'predictions' (majority class)

predictions = []
for ele in tokenized_data_test["labels"]:
    predictions.append(labels_to_ids["ASSERTIVE"]) # for coarse-grained classification
    # predictions.append(labels_to_ids["ASSERT"]) # for fine-grained classification

#### Print results as latex table

report = classification_report(tokenized_data_test["labels"], predictions, target_names=sorted(unique_labels),output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report.to_latex())

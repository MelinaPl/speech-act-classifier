import pandas as pd
import json
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch

#### Check device

print(torch.has_mps)
device = torch.device('mps')

#### Load my dataset

with open("/Users/melinaplakidis/Documents/Sprachtechnologie_HA/data/annotations_with_text.json", "r", encoding="utf8") as f:
    data = json.load(f)


def sentence_iterator(data):
    for ele in data:
        name = ele
        tweet = data[ele]
        for value in tweet.values():
            sentences = value['sentences']
            for sentence in sentences:
                text = sentences[sentence]['text']
                stype = sentences[sentence]['stype']
                coarse = sentences[sentence]['coarse']
                fine = sentences[sentence]['fine']
                yield text, coarse, fine

labels_to_exclude = ["DISAGREE", "APOLOGIZE", "THANK", "GREET"] # occur < 10 times in the entire dataset

texts, coarses, fines = [], [], []
for text, coarse, fine in sentence_iterator(data):
    texts.append(text)
    coarses.append(coarse)
    if fine in labels_to_exclude:
        fines.append("EXCLUDED")
    elif coarse == "COMMISSIVE": # no fine-grained categories for commissives
        fines.append("COMMISSIVE")
    else:
        fines.append(fine)

df = pd.DataFrame({"text": texts, "labels": fines})
print(df)

#### Convert to Huggingface dataset

train_df, test_df = train_test_split(df, test_size=0.2, random_state=200)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#### Get labels

labels = [i for i in df['labels'].values.tolist()]
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
    predictions.append(labels_to_ids["ASSERT"])

#### Print results as latex table

report = classification_report(tokenized_data_test["labels"], predictions, target_names=list(sorted(unique_labels)),output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(df_report.to_latex())
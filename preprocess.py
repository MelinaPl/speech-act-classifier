import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import os
import wandb

### Check device
print(torch.has_mps)
device = torch.device('mps')
### Log in to wandb

### Class for metrics
class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
    
    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)
    
    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()
    
#### LOAD MY DATA

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

texts, coarses, fines = [], [], []
for text, coarse, fine in sentence_iterator(data):
    texts.append(text)
    coarses.append(coarse)
    fines.append(fine)
  
df = pd.DataFrame({"text": texts, "labels": coarses})#, "fines": fines})
print(df)

#### CONVERT TO HF DATASET

unsplitted_dataset = Dataset.from_pandas(df)
dataset = unsplitted_dataset.train_test_split(test_size=0.2, seed=200)
print(dataset)

### START PREPROCESSING 

labels = [i for i in df['labels'].values.tolist()]
unique_labels = set()

for lb in labels:
    if lb not in unique_labels:
        unique_labels.add(lb)
labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

print(unique_labels)
print(labels_to_ids)
print(ids_to_labels)
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer.to(device)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-german-cased", num_labels=6, id2label=ids_to_labels, label2id=labels_to_ids
)
model.to(device)


def preprocess_function(examples):
    text = examples["text"]
    int_labels = [labels_to_ids[lb] for lb in examples["labels"]]
    examples["labels"] = int_labels
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

tokenized_data_train = dataset["train"].map(preprocess_function, batched=True, remove_columns="text")
tokenized_data_test = dataset["test"].map(preprocess_function, batched=True, remove_columns="text")
#print(tokenized_data_test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

classification_report_metric = evaluate.load("bstrai/classification_report")
# clf_metrics = evaluate.combine([
#     evaluate.load('accuracy'), 
#     ConfiguredMetric(evaluate.load('f1'), average='macro'),
#     ConfiguredMetric(evaluate.load('precision'), average='macro'),
#     ConfiguredMetric(evaluate.load('recall'), average='macro')
# ])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    predictions_str = [ids_to_labels[idx] for idx in predictions]
    labels_str = [ids_to_labels[idx] for idx in labels]
    print(f"Predictions: {predictions_str}, Labels: {labels_str}")
    return classification_report_metric.compute(predictions=predictions, references=labels)#, average='micro')

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="speech-acts"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

training_args = TrainingArguments(
    output_dir="model_with_metrics_coarse",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to='wandb',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())

### Uncomment the following to test trained model on one sentence

# input_text = "Das ist sehr falsch."
# inputs = tokenizer(input_text, return_tensors="pt")
# model = AutoModelForSequenceClassification.from_pretrained("/Users/melinaplakidis/Documents/Sprachtechnologie_HA/my_awesome_model/checkpoint-194/", local_files_only=True)
# with torch.no_grad():
#     logits = model(**inputs).logits
# predicted_class_id = logits.argmax().item()
# print(model.config.id2label[predicted_class_id])
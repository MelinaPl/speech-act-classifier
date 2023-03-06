import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch

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

# print(torch.has_mps)
# device = torch.device('mps')
# #### LOAD MY DATA
model = AutoModelForSequenceClassification.from_pretrained("/Users/melinaplakidis/Documents/Sprachtechnologie_HA/my_awesome_model/checkpoint-194/", local_files_only=True)

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

# #### CONVERT TO HF DATASET

unsplitted_dataset = Dataset.from_pandas(df)
dataset = unsplitted_dataset.train_test_split(test_size=0.2)

labels = [i for i in df['labels'].values.tolist()]
unique_labels = set()

for lb in labels:
    if lb not in unique_labels:
        unique_labels.add(lb)
labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

def preprocess_function(examples):
    text = examples["text"]
    int_labels = [labels_to_ids[lb] for lb in examples["labels"]]
    examples["labels"] = int_labels
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

tokenized_data_train = dataset["train"].map(preprocess_function, batched=True, remove_columns="text")
tokenized_data_test = dataset["test"].map(preprocess_function, batched=True, remove_columns="text")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

clf_metrics = evaluate.combine([
    evaluate.load('accuracy'), 
    ConfiguredMetric(evaluate.load('f1'), average='macro'),
    ConfiguredMetric(evaluate.load('precision'), average='macro'),
    ConfiguredMetric(evaluate.load('recall'), average='macro')
])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    predictions_str = [ids_to_labels[idx] for idx in predictions]
    labels_str = [ids_to_labels[idx] for idx in labels]
    print(f"Predictions: {predictions_str}, Labels: {labels_str}")
    return clf_metrics.compute(predictions=predictions, references=labels, average='micro')

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
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
print(trainer.evaluate())


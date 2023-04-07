import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import os
import wandb
from transformers import set_seed

#### Check device and set seed

set_seed(123)
print(torch.has_mps)
device = torch.device('mps')

#### Load my data

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
                if sentences[sentence]['coarse'] == "COMMISSIVE" or sentences[sentence]['coarse'] == "OTHER":
                    coarse = "COMOTH"
                else:
                    coarse = sentences[sentence]['coarse']
                fine = sentences[sentence]['fine']
                yield text, coarse, fine

texts, coarses, fines = [], [], []
for text, coarse, fine in sentence_iterator(data):
    texts.append(text)
    coarses.append(coarse)
    fines.append(fine)
  
df = pd.DataFrame({"text": texts, "labels": coarses})

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

#### Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-german-cased", num_labels=5, id2label=ids_to_labels, label2id=labels_to_ids
).to(device)


def preprocess_function(examples):
    text = examples["text"]
    int_labels = [labels_to_ids[lb] for lb in examples["labels"]]
    examples["labels"] = int_labels
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

tokenized_data_train = train_dataset.map(preprocess_function, batched=True, remove_columns="text")
tokenized_data_test = test_dataset.map(preprocess_function, batched=True, remove_columns="text")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions.argmax(-1)
    report = classification_report(labels, preds, target_names=sorted(unique_labels), zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.to_latex())
    return report


#### Set WANDB 

os.environ["WANDB_PROJECT"]="speech-acts"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

training_args = TrainingArguments(
    output_dir="coarse_cased_5classes",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to='wandb',
    use_mps_device=True,
    seed=123, 
    data_seed=123,
    full_determinism=True
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

#### Start training and evaluate

trainer.train()
print(trainer.evaluate())
wandb.finish()

### Uncomment the following to test trained model on one sentence

# input_text = "Das ist sehr falsch."
# inputs = tokenizer(input_text, return_tensors="pt")
# model = AutoModelForSequenceClassification.from_pretrained("/Users/melinaplakidis/Documents/Sprachtechnologie_HA/my_awesome_model/checkpoint-194/", local_files_only=True)
# with torch.no_grad():
#     logits = model(**inputs).logits
# predicted_class_id = logits.argmax().item()
# print(model.config.id2label[predicted_class_id])
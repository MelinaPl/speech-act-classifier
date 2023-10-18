import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
import torch
import os
import numpy as np
from transformers import set_seed
import wandb
import platform

#### Ensure reproducibility
os.environ['PYTHONHASHSEED']= "123"
set_seed(123)

#### check whether x86 or arm64 is used, should be the latter
print(platform.platform()) 

"""
If its x86, type the following into the console:

$ CONDA_SUBDIR=osx-arm64 conda create -n env_name -c conda-forge
$ conda activate env_name
$ conda config --env --set subdir osx-arm64
$ conda install python=3.9
$ pip install -r requirements.txt

"""
### Start a new wandb run to track this script
wandb.init(
    project="speech-acts"
)

#### This ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
#### This ensures that the current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

#### Set device (only for Mac!)
device = torch.device("mps")

#### Set paths 

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = "data"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
full_path = os.path.join(WORKING_DIR, DATA_DIR)
print(full_path)

#### Load my dataset

train = pd.read_csv(os.path.join(full_path,"train_fine.csv"),sep='\t')
test = pd.read_csv(os.path.join(full_path,"test_fine.csv"), sep="\t")

#### Get label mappings

labels = [i for i in train['labels'].values.tolist()]
unique_labels = set()

for lb in labels:
    if lb not in unique_labels:
        unique_labels.add(lb)
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
print(unique_labels)
print(len(unique_labels))

#### Load tokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

#### Transform to HF datasets

train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)
print(train_dataset[0])

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
    macro_f1 = report["macro avg"]["f1-score"]
    weight_f1 = report["weighted avg"]["f1-score"]
    score_dict = {"macro_f1": macro_f1, "weighted_f1": weight_f1}
    return score_dict

#### Set WANDB 

os.environ["WANDB_PROJECT"]="speech-acts"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

def model_init():
    return AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-german-cased",
    num_labels = len(unique_labels), # The number of output labels
    id2label=ids_to_labels,
    label2id=labels_to_ids).to(device)

training_args = TrainingArguments(
    output_dir="model_name",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    optim="adamw_torch",
    save_total_limit=1,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    data_seed=123,
    seed=123,
    full_determinism=True,
    use_mps_device=True
)

trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#### Start training and evaluate

trainer.train()
print(trainer.evaluate())

#### After training, access the path of the best checkpoint like this
best_ckpt_path = trainer.state.best_model_checkpoint
print(best_ckpt_path)
wandb.finish()

# ### Uncomment the following to test trained model on one sentence

# # input_text = "Das ist sehr falsch."
# # inputs = tokenizer(input_text, return_tensors="pt")
# # model = AutoModelForSequenceClassification.from_pretrained("Path/to/model/checkpoint-194/", local_files_only=True)
# # with torch.no_grad():
# #     logits = model(**inputs).logits
# # predicted_class_id = logits.argmax().item()
# # print(model.config.id2label[predicted_class_id])
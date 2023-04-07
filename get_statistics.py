import pandas as pd
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch

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
                coarse = sentences[sentence]['coarse']
                fine = sentences[sentence]['fine']
                yield text, coarse, fine

texts, coarses, fines = [], [], []
for text, coarse, fine in sentence_iterator(data):
    texts.append(text)
    coarses.append(coarse)
    fines.append(fine)
  
df = pd.DataFrame({"text": texts, "coarses": coarses, "fines": fines})

#### Convert to Huggingface dataset and split into train and test set

train_df, test_df = train_test_split(df, test_size=0.2, random_state=200) # set random state to ensure reproducible results
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#### Count instances for each class (coarse-grained) in train and test set

train_dict, test_dict = {}, {}
train_total, test_total = 0, 0
for label in train_dataset["coarses"]:
    if not label in train_dict:
        train_dict[label] = 1
    else:
        train_dict[label] += 1
    train_total += 1
train_dict["Total"] = train_total
print(train_dict)

for label in test_dataset["coarses"]:
    if not label in test_dict:
        test_dict[label] = 1
    else:
        test_dict[label] += 1
    test_total += 1
test_dict["Total"] = test_total
print(test_dict)

#### Save in dataframe and print latex table for coarse-grained speech acts

coarse_df = pd.DataFrame({"Train" : train_dict,"Test": test_dict})
print(coarse_df.transpose().to_latex())


#### Count instances for each class (fine-grained) in train and test set

train_dict, test_dict = {}, {}
train_total, test_total = 0, 0
for fine in train_dataset["fines"]:
    if not fine in train_dict:
        train_dict[fine] = 1
    else:
        train_dict[fine] += 1
    train_total += 1
train_dict["Total"] = train_total
print(train_dict)

for fine in test_dataset["fines"]:
    if not fine in test_dict:
        test_dict[fine] = 1
    else:
        test_dict[fine] += 1
    test_total += 1
test_dict["Total"] = test_total
print(test_dict)

#### Save in dataframe and print latex table for fine-grained speech acts

fine_df = pd.DataFrame({"Train" : train_dict,"Test": test_dict})
print(fine_df.to_latex())
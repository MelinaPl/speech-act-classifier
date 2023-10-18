import pandas as pd
import json
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import os


#### Set device (for Mac)

print(torch.has_mps)
device = torch.device('mps')


def sentence_iterator_coarse_merged(data):
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

def sentence_iterator_coarse(data):
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

def sentence_iterator_fine(data):
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

def create_csv_files(version, path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if version == "fine":
        print("WORKS")
        labels_to_exclude = ["DISAGREE", "APOLOGIZE", "THANK", "GREET"] # occur < 10 times in the entire dataset

        texts, coarses, fines = [], [], []
        for text, coarse, fine in sentence_iterator_fine(data):
            texts.append(text)
            coarses.append(coarse)
            if fine in labels_to_exclude:
                fines.append("EXCLUDED")
            elif coarse == "COMMISSIVE": # no fine-grained categories for commissives
                fines.append("COMMISSIVE")
            else:
                fines.append(fine)
  
        df = pd.DataFrame({"text": texts, "labels": fines})
        return df
    
    elif version == "coarse":
        texts, coarses, fines = [], [], []
        for text, coarse, fine in sentence_iterator_coarse(data):
            texts.append(text)
            coarses.append(coarse)
            fines.append(fine)
  
        df = pd.DataFrame({"text": texts, "labels": coarses})
        return df

    elif version == "coarse_merged":
        texts, coarses, fines = [], [], []
        for text, coarse, fine in sentence_iterator_coarse_merged(data):
            texts.append(text)
            coarses.append(coarse)
            fines.append(fine)
  
        df = pd.DataFrame({"text": texts, "labels": coarses})
        return df

if __name__ == "__main__":
    version_name = "fine" # Possible other values: 'coarse', 'coarse_merged'
    pathname = "Path/to/version_1-1.json" # Set to location of file: https://github.com/MelinaPl/speech-act-analysis/blob/main/data/version_1-1.json
    df = create_csv_files(version=version_name, path=pathname)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=200)
    train_df.to_csv("data/train_"+ str(version_name)+".csv", sep='\t', encoding='utf-8', index=False)
    test_df.to_csv("data/test_"+ str(version_name)+".csv", sep='\t', encoding='utf-8', index=False)
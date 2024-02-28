import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import spacy
from spacymoji import Emoji

pd.options.mode.chained_assignment = None 

#### Set paths 

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = "data"
full_path = os.path.join(WORKING_DIR, DATA_DIR)

#### Features 

def add_punctuation_feature(text):
    count = 0
    for token in text:
        if token.is_punct:
            count += 1
    return count

def add_exclamation(text):
    one_hot_value = 0
    for token in text:
        if token.text == "!":
            one_hot_value = 1
    return one_hot_value

def add_punctuation_mark(text):
    one_hot_value = 0
    for token in text:
        if token.text == ".":
            one_hot_value = 1
    return one_hot_value

def add_question_mark(text):
    one_hot_value = 0
    for token in text:
        if token.text == "?":
            one_hot_value = 1
    return one_hot_value

def add_hashtag(text):
    one_hot_value = 0
    for token in text:
        if "#" in token.text:
            one_hot_value = 1
    return one_hot_value

def add_mention(text):
    one_hot_value = 0
    for token in text:
        if "@" in token.text:
            one_hot_value = 1
    return one_hot_value

def add_capital_char_count(text):
    count=0
    for token in text:
        if token.text.isupper():
            count+=1
    return count

def add_token_count(text):
    count = 0
    for token in text:
        count += 1
    return count

def add_char_length(text):
    return len(text.text)

def add_emoji_count(text):
    count = 0
    for token in text:
        if token._.is_emoji:
            count += 1
    return count


def add_emoji_onehot(text):
    if text._.has_emoji == True:
        return 1
    else:
        return 0
    
def main_features(df, nlp):
    punct_count, contains_exclamation, contains_punct_mark, contains_question_mark, contains_hashtag = [], [], [], [], []
    contains_mention, capital_char_count, token_count, char_length, emoji_count, contains_emoji = [], [], [], [], [], []
    for text in df["text"]:
        doc = nlp(text)
        punct_count.append(add_punctuation_feature(doc))
        contains_exclamation.append(add_exclamation(doc))
        contains_punct_mark.append(add_punctuation_mark(doc))
        contains_question_mark.append(add_question_mark(doc))
        contains_hashtag.append(add_hashtag(doc))
        contains_mention.append(add_mention(doc))
        capital_char_count.append(add_capital_char_count(doc))
        token_count.append(add_token_count(doc))
        char_length.append(add_char_length(doc))
        emoji_count.append(add_emoji_count(doc))
        contains_emoji.append(add_emoji_onehot(doc))
    df['punct_count'] = punct_count
    df['contains_exclamation'] = contains_exclamation
    df['contains_punct_mark'] = contains_punct_mark
    df['contains_question_mark'] = contains_question_mark
    df['contains_hashtag'] = contains_hashtag
    df['contains_mention'] = contains_mention
    df['capital_char_count'] = capital_char_count
    df['token_count'] = token_count
    df['char_length'] = char_length
    df['emoji_count'] = emoji_count
    df['contains_emoji'] = contains_emoji
    return df


if __name__ == '__main__':   

    #### Load my dataset

    train = pd.read_csv(os.path.join(full_path,"train_coarse.csv"),sep='\t')
    test = pd.read_csv(os.path.join(full_path,"test_coarse.csv"), sep="\t")
    # train = pd.read_csv(os.path.join(full_path,"train_fine.csv"),sep='\t')
    # test = pd.read_csv(os.path.join(full_path,"test_fine.csv"), sep="\t")
    # train = pd.read_csv(os.path.join(full_path,"train_coarse_merged.csv"),sep='\t')
    # test = pd.read_csv(os.path.join(full_path,"test_coarse_merged.csv"), sep="\t")

    #### Load spacy model

    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("emoji", first=True)

    #### Transform labels to numbers

    Encoder = LabelEncoder()
    labels = Encoder.fit(train["labels"])
    Y_train = labels.transform(train["labels"])
    Y_test = labels.transform(test["labels"])
    target_names = Encoder.classes_
    label_indices = Encoder.transform(Encoder.classes_)

    train = main_features(train, nlp)
    test = main_features(test, nlp)

    #### Transform words to lower and lemmatize

    train["text"] = [text.lower() for text in train["text"]]
    test["text"] = [text.lower() for text in test["text"]]

    for i, text in enumerate(train["text"]):
        doc = nlp(text)
        lemmas = ' '.join([token.lemma_ for token in doc if not token.is_punct])
        train["text"][i] = lemmas

    for i, text in enumerate(test["text"]):
        doc = nlp(text)
        lemmas = ' '.join([token.lemma_ for token in doc if not token.is_punct])
        test["text"][i] = lemmas

    #### Vectorize text
        
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 2), lowercase=False
    )
    X_train= vectorizer.fit_transform(train["text"]).toarray()
    X_test= vectorizer.transform(test["text"]).toarray()
    train_tf_idf = pd.DataFrame(X_train).add_prefix('tf_idf_')
    test_tf_idf  = pd.DataFrame(X_test).add_prefix('tf_idf_')

    #### Finally merging all features with above TF-IDF. 

    features = list(train.columns.values)
    features = [feature for feature in features if not (feature == "text" or feature == "labels")]
    X_train_features = pd.merge(train_tf_idf,train[features],left_index=True, right_index=True)
    X_test_features  = pd.merge(test_tf_idf,test[features],left_index=True, right_index=True)

    #### Train Classifier

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', random_state=123)

    clf_nofeat = SVM.fit(X_train, Y_train)
    predictions_SVM_nofeat = clf_nofeat.predict(X_test)
    print("--------------NO FEATURES---------------")
    print(classification_report(Y_test, predictions_SVM_nofeat, labels=label_indices, target_names=target_names))

    clf = SVM.fit(X_train_features.values, Y_train)
    predictions_SVM = clf.predict(X_test_features)
    print("--------------WITH FEATURES---------------")
    print(classification_report(Y_test, predictions_SVM, labels=label_indices, target_names=target_names))
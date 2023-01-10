import os
os.chdir(r"c:\Users\britt\Desktop\YH\Applicerad AI\job_discrimination")
import re

import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from unidecode import unidecode


def read_file(file_name):
    """
    This function will read the text files passed & return the list
    """
    with open(file_name, "r", encoding="utf-8") as f:
        words = f.read().replace("\n", " ")

    return words


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return ""


def penn_to_wn(tag):
    return get_wordnet_pos(tag)


def clean_text(corpus):
    stemmer = WordNetLemmatizer()
    preprocessed_corpus = []

    for i, document in enumerate(corpus):
        remove_https = re.sub(r"http\S+", "", document)
        remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
        remove_numbers_punctuations = re.sub(r"[^a-zA-Z]+", " ", remove_com) 
        pattern = re.compile(r'\s+') 
        remove_extra_whitespaces = re.sub(pattern, " ", remove_numbers_punctuations)
        only_ascii = unidecode(remove_extra_whitespaces)
        doc = only_ascii.lower()

        list_of_tokens = word_tokenize(doc)
        list_of_tokens_pos = pos_tag(list_of_tokens)
        list_of_tokens_wn_pos = [(token[0], penn_to_wn(token[1])) for token in list_of_tokens_pos if token[0] not in stopwords.words("english")]
        list_of_lemmas = [stemmer.lemmatize(token[0], token[1]) if token[1] != "" else stemmer.lemmatize(token[0]) for token in list_of_tokens_wn_pos]
        
        preprocessed_corpus.append(" ".join(list_of_lemmas))
    
    return preprocessed_corpus


def add_labels_to_df(majority_share=0.7):
    df = pd.read_csv("data/cleaned_data/applicants.csv", dtype={'ID': object})  

    mostly_women = df["Female"] >= (df["Apps Received"] - df["Unknown_Gender"]) * majority_share
    mostly_men = df["Male"] >= (df["Apps Received"] - df["Unknown_Gender"]) * majority_share

    labels = []
    label_ints = []
    for i in range(len(df)):
        if mostly_women[i]:
            label = "W"
            label_int = 1
        elif mostly_men[i]:
            label = "M"
            label_int = 2
        else:
            label = "N"
            label_int = 0
        labels.append(label)
        label_ints.append(label_int)
    
    majority_percent = majority_share * 10
    minority_percent = 100 - majority_percent
    df[f"Label {majority_percent}/{minority_percent}"] = labels
    df[f"Numeric label {majority_percent}/{minority_percent}"] = label_ints


def extract_ngrams(df):
    bigrams = []
    trigrams = []
    cleaned_texts = list(df["Cleaned text"])
    for text in cleaned_texts:
        bigrams.append(list(ngrams(text.split(), 2)))
        trigrams.append(list(ngrams(text.split(), 3)))


def get_n_most_important_words(weights, vocabulary, n):
    indices = np.argpartition(weights, len(weights) - n)[-n:]
    min_elements = weights[indices]
    min_elements_order = np.argsort(min_elements)
    ordered_indices = indices[min_elements_order]
    words = [vocabulary[i] for i in ordered_indices]

    return words

import os
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import glob
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer


def get_contents(file_paths) -> list:
    """Create list of contents from all files in a list of file paths"""
    contents = []
    file_names = []
    for file_path in file_paths:
        with open(file_path) as f:
            # Remove newlines
            content = f.read().replace("\n", " ")
            # Remove numbers
            content = re.sub("\d", "", content)
        contents.append(content)
        file_names.append(os.path.basename(file_path))

    return file_names, contents

def count_word_freq(text_sample, words) -> DataFrame:
    # Create vector with count of words in the text sample
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text_sample])
    df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Create iterable with all words in the text sample
    words_in_doc = set(df_bow.columns.values)

    # Create a new list with the words that are common between words_in_doc and words
    words_in_common = list(words_in_doc.intersection(words))
    df_words = df_bow[words_in_common]
    return df_words

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

def lemmatize(corpus):
    lemmatizer = WordNetLemmatizer()
    preprocessed_corpus = []

    for i, document in enumerate(corpus):
        remove_https = re.sub(r"http\S+", "", document)
        remove_com = re.sub(r"\ [A-Za-a]*\.com", " ", remove_https)
        remove_numbers_punctuations = re.sub(r"[^a-zA-Z]+", " ", remove_com)
        pattern = re.compile(r"\s+")
        remove_extra_whitespaces = re.sub(pattern, " ", remove_numbers_punctuations)
        only_ascii = unidecode(remove_extra_whitespaces)
        doc = only_ascii.lower()

        list_of_tokens = word_tokenize(doc)
        list_of_tokens_pos = pos_tag(list_of_tokens)
        list_of_tokens_wn_pos = [(token[0], penn_to_wn(token[1])) for token in list_of_tokens_pos if token[0] not in stopwords.words("english")]
        list_of_lemmas = [lemmatizer.lemmatize(token[0], token[1]) if token[1] != "" else lemmatizer.lemmatize(token[0]) for token in list_of_tokens_wn_pos]

        preprocessed_corpus.append(" ".join(list_of_lemmas))
    return preprocessed_corpus
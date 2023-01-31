import os
import re

import gensim
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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


def count_word_freq(text_sample, words) -> pd.DataFrame:
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


def preprocess_text(corpus):
    lemmatizer = WordNetLemmatizer()
    preprocessed_corpus = []

    for _, document in enumerate(corpus):
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


# Den här är för att predicta en text, ska till finals
def preprocess_document(doc):
    lemmatizer = WordNetLemmatizer()

    remove_https = re.sub(r"http\S+", "", doc)
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

    return list_of_lemmas


def read_file(file_name):
    """
    This function will read the text files passed & return the list
    """
    with open(file_name, "r", encoding="utf-8") as f:
        words = f.read().replace("\n", " ")

    return words


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


def get_n_most_important_words_clf(weights, vocabulary, n):
    indices = np.argpartition(weights, len(weights) - n)[-n:]
    min_elements = weights[indices]
    min_elements_order = np.argsort(min_elements)
    ordered_indices = indices[min_elements_order]
    words = [vocabulary[i] for i in ordered_indices]
    weights = [round(weights[i], 5) for i in ordered_indices]

    return words[::-1], weights[::-1]


def get_25_most_important_words_single_text(text, label, clf_vocabulary, clf_weights):
    vocabulary_text = list(set(text))
    text_weights = []
    words, weights = get_n_most_important_words_clf(clf_weights[label], clf_vocabulary, len(clf_vocabulary))
    weights_dict = dict(zip(words, weights))
    for word in vocabulary_text:
        try:
            text_weights.append((word, weights_dict[word]))
        except KeyError:
            pass
    
    text_weights.sort(key=lambda x: x[1], reverse=True)
    
    return text_weights[:25]


def corpus_to_doc_vectors():
    df = pd.read_csv("data/cleaned_data/bulletins_labels_share_content.csv", dtype={'ID': object})
    corpus = list(df["Cleaned text"])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(corpus)
    google_model = gensim.models.KeyedVectors.load_word2vec_format("c:/Users/britt/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

    vocabulary = tfidf_vectorizer.get_feature_names_out()
    documents_embeddings = []
    documents_scaled_embeddings = []
    for doc in corpus:
        word_embeddings = []
        scaled_embeddings  = []
        doc_list = doc.split()
        for word in doc_list:
            if word in google_model.key_to_index.keys():
                embedding = google_model[word]
                word_embeddings.append(embedding)
                index = np.where(vocabulary == word)[0]
                try:
                    scaled_embeddings.append(embedding * tfidf_vectorizer.idf_[index])
                except ValueError:
                    pass
        documents_embeddings.append(word_embeddings)
        documents_scaled_embeddings.append(scaled_embeddings)

        doc_vectors = [np.average(doc, axis=0) for doc in documents_embeddings]
        scaled_doc_vectors = [np.average(doc, axis=0) for doc in documents_scaled_embeddings]

        return doc_vectors


def job_ad_to_doc_vector_filename(file_name):
    df = pd.read_csv("data/cleaned_data/bulletins_labels_share_content.csv", dtype={'ID': object})
    corpus = list(df["Cleaned text"])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(corpus)
    google_model = gensim.models.KeyedVectors.load_word2vec_format("c:/Users/britt/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

    with open(f"data/original_data/job_bulletins/{file_name}", "r", encoding="utf-8") as f:
        application = f.read()

    text = preprocess_document(application)
    regressor_vocabulary = tfidf_vectorizer.get_feature_names_out()

    scaled_embeddings  = []
    doc_list = text[0].split()
    for word in doc_list:
        if word in google_model.key_to_index.keys():
            embedding = google_model[word]
            index = np.where(regressor_vocabulary == word)[0]
            try:
                scaled_embeddings.append(embedding * tfidf_vectorizer.idf_[index])
            except ValueError:
                pass

    doc_vector = np.average(scaled_embeddings, axis=0)

    return doc_vector


def job_ad_to_doc_vector_text(text):
    df = pd.read_csv("data/cleaned_data/bulletins_labels_share_content.csv", dtype={'ID': object})
    corpus = list(df["Cleaned text"])
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(corpus)
    google_model = gensim.models.KeyedVectors.load_word2vec_format("c:/Users/britt/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

    regressor_vocabulary = tfidf_vectorizer.get_feature_names_out()

    scaled_embeddings  = []
    doc_list = text[0].split()
    for word in doc_list:
        if word in google_model.key_to_index.keys():
            embedding = google_model[word]
            index = np.where(regressor_vocabulary == word)[0]
            try:
                scaled_embeddings.append(embedding * tfidf_vectorizer.idf_[index])
            except ValueError:
                pass

    doc_vector = np.average(scaled_embeddings, axis=0)

    return doc_vector


def predict_important_words(text):
    with open("data/models/log_reg_tfidf.pkl", "rb") as read_file:
        clf = pickle.load(read_file)

    with open("data/vectorizers/tfidf.pkl", "rb") as read_file:
        vectorizer = pickle.load(read_file)

    clf_weights = clf.coef_
    clf_vocabulary = vectorizer.get_feature_names_out()

    vectorized_text = vectorizer.transform(text)
    prediction = clf.predict(vectorized_text)

    predicted_label = prediction[0]
    top_25_words = get_25_most_important_words_single_text(text, predicted_label, clf_vocabulary, clf_weights)

    return top_25_words


def predict_label(doc_vector):
    with open("data/models/log_reg_scaled_emb.pkl", "rb") as read_file:
        clf = pickle.load(read_file)

    predicted_label = clf.predict(doc_vector.reshape(1, -1))[0]
    pred_probas = clf.predict_proba(doc_vector.reshape(1, -1))

    return predicted_label, pred_probas


def predict_distribution(doc_vector):
    with open("data/models/cat_boost_regr.pkl", "rb") as read_file:
        regressor = pickle.load(read_file)

    cat_boost_prediction = regressor.predict(doc_vector)

    return cat_boost_prediction


def create_result_dict(file_name, id, title, words, predicted_label, pred_probas, distribution):
    label_to_word = {
        0: ("Neutral", "30-69% male/female applicants"),
        1: ("Female", "More than 70 percent female applicants"),
        2: ("Male", "More than 70% male applicants")
    }

    result = {
        "File name": file_name, 
        "Id": str(id),
        "Title": title,
        "Top 25 important words": words,
        "Predicted numeric label": str(predicted_label),
        "Predicted label word": label_to_word[predicted_label][0],
        "Predicted label meaning": label_to_word[predicted_label][1],
        "Probability label neutral": f"{round(pred_probas[0][0] * 100, 1)}%",
        "Probability label female": f"{round(pred_probas[0][1] * 100, 1)}%",
        "Probability label male": f"{round(pred_probas[0][2] * 100, 1)}%",
        "Predicted male distribution": f"{round(distribution[0] * 100, 1)}%",
        "Predicted female distribution": f"{round(distribution[1] * 100, 1)}%"
    }

    return result
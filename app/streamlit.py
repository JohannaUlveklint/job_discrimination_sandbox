import streamlit as st
import pandas as pd
import warnings
import numpy as np
import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from io import StringIO


# https://www.youtube.com/watch?v=-IM3531b1XU&ab_channel=M%C4%B1sraTurp

st.markdown(
    """
    <style>
    .main {
    background-color: #b6d7a8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    df = pd.read_csv(filename)
    return df

load_file = st.container() 
models = st.container()
interpretation = st.container()

with load_file:
    st.header("Distribution of applicants")
    st.subheader("Start by uploading your job bulletin")
    upload_col, ok_col = st.columns(2)
    user_file = upload_col.file_uploader(label="**Upload your .txt-file here**", type="txt") 
    if user_file is not None:
        # To read file as bytes
        bytes_data = user_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(user_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write("**You uploaded this file:**")
        st.write(f"{string_data[:500]}...")

        # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(user_file)
        # st.write(dataframe)

        # Also solution for multiple uploads at https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader

    



with models:
    class_col, reg_col = st.columns(2)

    # Classification model
    class_col.header("Results from classification model")
    warnings.simplefilter("ignore")
    # df = pd.read_csv("../data/cleaned_data/bulletins_labels_share_content.csv")
    df = get_data("../data/cleaned_data/bulletins_w_labels_and_content.csv")
    

    corpus = list(df["Cleaned text"])
    google_model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/Johanna/Downloads/archive_word_vectors/GoogleNews-vectors-negative300.bin.gz", binary=True)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(corpus)

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

    df["Scaled embeddings"] = documents_scaled_embeddings
    scaled_doc_vectors = [np.average(doc, axis=0) for doc in df["Scaled embeddings"]]
    X_scaled = np.array(scaled_doc_vectors)
    y = df["Numeric label 70/30"]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

    clf = LogisticRegression(C=8.091237124889124, class_weight="balanced")
    clf.fit(X_train, y_train)


    # Regression model
    reg_col.header("Results from regression model")


with interpretation:
    col_1, col_2, col_3 = st.columns(3)
    col_1.subheader("Interpretated result:")
    col_1.markdown("* **The models suggests that more men will apply for this job**")
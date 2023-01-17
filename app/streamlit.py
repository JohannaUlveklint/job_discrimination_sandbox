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
    class_col.subheader("Results from classification model")
    # https://www.youtube.com/watch?v=A4K6D_gx2Iw&ab_channel=sentdex
    CATEGORIES = ["neutral", "female", "male"]

    def prepare(filepath):
        # Do something with the .txt-file
        return None

    @st.cache
    def get_model():
        return "Model saved somewhere"
    
    model = get_model()
    # prediction = model.predict([prepare(string_data)])  # Maybe not string_data, maybe not in a list
    # prediction = CATEGORIES[int(prediction[0][0])]
    pred_class = "female"
    class_col.write(f"The classification model predicts that the distribution of applicants will be mostly \033[1m{pred_class}\033[0m.")

    # Regression model
    reg_col.subheader("Results from regression model")
    pred_reg = 0.44
    reg_col.write(f"The regression model predicts that \033[1m{round((1 - pred_reg) * 100, 1)}% women\033[0m and \033[1m{pred_reg * 100}% men\033[0m will apply for this job.")


with interpretation:
    col_1, col_2, col_3 = st.columns(3)
    col_2.subheader("Interpretated result:")
    if pred_class == "female":
        col_2.markdown(f"* The classification model predicts that the distribution will be mostly \033[1m{pred_class}\033[0m.")
        col_2.markdown(f"* The regression model predicts that {round((1 - pred_reg) * 100, 1)} wommen will apply for this job.")
        col_2.markdown(f"* Due to a low share of jobs that mostly women applies to, the numbers from the regression model is probably misleading.")
        col_2.markdown(f"* More correct numbers would be that over 70% of the applicants will be women.")
    else:
        col_2.markdown(f"* The models predicts that the applicants for this job will be mostly {pred_class}, "\
        f"{round((1 - pred_reg) * 100, 1)}% women and {pred_reg * 100}% men.")

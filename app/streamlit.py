import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn import tree, preprocessing
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

load_file = st.container()  # OK button here or in own container?
models = st.container()
interpretation = st.container()

with load_file:
    st.header("Distribution of applicants")
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

    # df = pd.read_csv("../data/cleaned_data/bulletins_labels_share_content.csv")
    df = get_data("../data/cleaned_data/bulletins_w_labels_and_content.csv")
    st.write(df.head())

    st.subheader("Apps Received")
    apps_received = pd.DataFrame(df["Apps Received"].value_counts())
    st.bar_chart(apps_received)


with models:
    class_col, reg_col = st.columns(2)

    # Classification model
    class_col.header("Results from classification model")
    X = df["Cleaned text"]
    y = df["Numeric label 70/30"]
    vect = CountVectorizer(stop_words="english")
    X = vect.fit_transform(X).todense()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1000)
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train.astype('int'))
    y_pred = model.predict(X_test)

    electrical_repairer = "../data/cleaned_data/Job_Bulletins/unlabeled/ELECTRICAL REPAIRER 3853"


    # Regression model
    reg_col.header("Results from regression model")


with interpretation:
    st.subheader("Interpretated result:")
    st.markdown("* **The model suggests that more men will apply for this job**")
import pickle

import src.helpers as helpers

print("Reading file...")
with open("data/cleaned_data/Job_Bulletins/unlabeled/GALLERY ATTENDANT 2442", "r", encoding="utf-8") as ad_file:
    ad_text = ad_file.read()

# with open("data/models/log_reg_tfidf.pkl", "rb") as read_file:
#         clf = pickle.load(read_file)

# with open("data/vectorizers/tfidf.pkl", "rb") as read_file:
#         vectorizer = pickle.load(read_file)

print("Cleaning text...")
cleaned_text = helpers.preprocess_document(ad_text)
# vocabulary_text = list(set(cleaned_text))
# print(vocabulary_text)

print("Converting ad to document embedding...")
doc_vector = helpers.job_ad_to_doc_vector_text(cleaned_text)
print(doc_vector)

# clf_weights = clf.coef_
# clf_vocabulary = vectorizer.get_feature_names_out()

# vectorized_text = vectorizer.transform(cleaned_text)
# prediction = clf.predict(vectorized_text)

# predicted_label = prediction[0]

# words = helpers.get_25_most_important_words_single_text(cleaned_text, predicted_label, clf_vocabulary, clf_weights)

# print(words)
# print(predicted_label)
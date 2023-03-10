#!/usr/bin/env python
# encoding: utf-8
import json

from flask import Flask, request, jsonify
import gensim

import src.helpers as helpers

#google_model = gensim.models.KeyedVectors.load_word2vec_format("c:/Users/britt/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

app = Flask(__name__)


@app.route('/upload_json', methods=['POST'])
def post_job_ad():
    ad_json = json.loads(request.data)
    text = ad_json["Text"]
    cleaned_text = helpers.preprocess_document(text)
    top_25_words = helpers.predict_important_words(cleaned_text)
    doc_vector = helpers.job_ad_to_doc_vector_text(cleaned_text)
    predicted_label, pred_probas = helpers.predict_label(doc_vector)
    predicted_distribution = helpers.predict_distribution(doc_vector)
    result = helpers.create_result_dict(ad_json["File name"], ad_json["Id"], ad_json["Title"], top_25_words, 
                                        predicted_label, pred_probas, predicted_distribution)
    
    return jsonify(result)
    

if __name__ == '__main__':
    app.run(debug=True)
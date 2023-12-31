import requests
from flask import Flask, request, app, json, jsonify, url_for, render_template
import pickle
import pandas as pd


app = Flask(__name__)


tfidf_path = 'notebook/tfidf.pkl'
model_path = 'notebook/my_model.pkl'

# Load the vectorizer
with open(tfidf_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        data = request.get_json()
        text = data['text']

        transformed_text = vectorizer.transform([text])

        prediction = model.predict(transformed_text.toarray())

        return jsonify({"response":"Hello"})
    else:
        return jsonify({"error": "method not allowed"})



if __name__ == "__main__":
    app.run(debug = False)


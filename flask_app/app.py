from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient


# 1. ROBUST NLTK SETUP
resources = ['stopwords', 'wordnet', 'omw-1.4']
for resource in resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


# 2. PREPROCESSING FUNCTIONS (Fixed Logic Order)
def lower_case(text):
    return text.lower()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def removing_punctuations(text):
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def normalize_text(text):
    # 1. Clean noise first
    text = lower_case(text)
    text = removing_urls(text)
    text = removing_punctuations(text) # Important: remove punct BEFORE stop words
    text = removing_numbers(text)
    
    # 2. Semantic processing
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text


# 3. MLFLOW & DAGSHUB SETUP
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ThE-GuY-sHuBhAm"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ThE-GuY-sHuBhAm"
repo_name = "dvc_ml_pipeline"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# 4. LOAD MODEL & VECTORIZER (From Registry Backup)
print("⏳ Connecting to Model Registry...")
client = MlflowClient()


try:
    latest_versions = client.get_latest_versions("model", stages=["Staging"])
    if not latest_versions:
        # Fallback to None only if Staging is empty
        latest_versions = client.get_latest_versions("model", stages=["None"])
    
    if not latest_versions:
        raise Exception("No models found in Registry.")

    run_id = latest_versions[0].run_id
    print(f"Downloading artifacts from Run ID: {run_id}...")

    
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path="model_backup"
    )
    
    # Load Model
    model_path = os.path.join(local_path, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Load Vectorizer
    vec_path = os.path.join(local_path, "vectorizer.pkl")
    if not os.path.exists(vec_path):
        print("Vectorizer not found in run. Attempting local fallback...")
        vec_path = 'ml-pipeline-imdb-movies-review/models/vectorizer.pkl'
        
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    print("Model and Vectorizer loaded successfully!")

except Exception as e:
    print(f"Critical Error Loading Model: {e}")


@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        original_text = request.form['text']
        
        # 1. Preprocess
        clean_text = normalize_text(original_text)
        
        # DEBUG LOGS
        print("\n" + "="*30)
        print(f"INPUT: {original_text}")
        print(f"CLEAN: {clean_text}")

        # 2. Vectorize
        features = vectorizer.transform([clean_text])
        
        # DEBUG LOGS
        print(f"VECTOR SUM: {features.sum()}")
        if features.sum() == 0:
            print("⚠️ WARNING: Vector is empty! Prediction will be default (Positive).")

        # 3. Predict (Using numpy array directly)
        prediction = model.predict(features.toarray())[0]
        proba = model.predict_proba(features.toarray())[0]
        
        print(f"PREDICTION: {prediction}")
        print(f"PROBABILITY: {proba}") # e.g., [0.1, 0.9]
        print("="*30 + "\n")

        return render_template('index.html', result=prediction)
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
import mlflow
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging
import dagshub
import os
import mlflow.sklearn

# --------------------------------------------------
# 1. SETUP 
# --------------------------------------------------
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "ThE-GuY-sHuBhAm"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ThE-GuY-sHuBhAm"
repo_name = "dvc_ml_pipeline"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
mlflow.autolog(disable=True)

# --------------------------------------------------
# 2. LOGGING CONFIG
# --------------------------------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_data(file_path: str):
    return pd.read_csv(file_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics, y_pred

def main():
    mlflow.set_experiment("dvc-pipeline-imdb-movies-review")

    with mlflow.start_run() as run:
        try:
            logger.info("Loading model and data...")
            model_path = 'ml-pipeline-imdb-movies-review/models/model.pkl'
            # Define path for vectorizer
            vectorizer_path = 'ml-pipeline-imdb-movies-review/models/vectorizer.pkl'
            
            data_path = 'ml-pipeline-imdb-movies-review/data/processed/test_tfidf.csv'
            
            clf = load_model(model_path)
            test_data = load_data(data_path)

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics, y_pred = evaluate_model(clf, X_test, y_test)

            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)

            metrics_path = 'ml-pipeline-imdb-movies-review/reports/metrics.json'
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact(metrics_path)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # --------------------------------------------------
            # 6. LOG MODEL & VECTORIZER
            # --------------------------------------------------
            logger.info("Starting Artifact Upload...")
            
            # Standard MLflow Model log (still keeping this for metadata, even if upload fails on Windows)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                serialization_format="cloudpickle",
                pip_requirements=[],
                metadata={"model_type": "GradientBoostingClassifier"}
            )
            
            # Upload Raw Model Pickle (Backup)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model_backup")
                logger.info("Raw model pickle uploaded.")
            
            # Upload Raw Vectorizer Pickle (The Upgrade!)
            if os.path.exists(vectorizer_path):
                mlflow.log_artifact(vectorizer_path, artifact_path="model_backup")
                logger.info("Raw vectorizer pickle uploaded.")
            else:
                logger.warning(f"Vectorizer not found at {vectorizer_path}. Did you run feature_engineering?")
            
            logger.info("All artifacts uploaded successfully.")

            info_path = 'ml-pipeline-imdb-movies-review/reports/experiment_info.json'
            with open(info_path, 'w') as f:
                json.dump({'run_id': run.info.run_id, 'model_path': 'model'}, f, indent=4)
            
            logger.info(f"Run ID {run.info.run_id} saved to {info_path}")

        except Exception as e:
            logger.error(f"Failed during run: {e}")
            raise

if __name__ == '__main__':
    main()
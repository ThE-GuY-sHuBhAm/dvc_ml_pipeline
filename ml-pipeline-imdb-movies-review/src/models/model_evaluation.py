import mlflow
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging
import dagshub
import json
import os



dagshub.init(repo_owner="ThE-GuY-sHuBhAm", repo_name="dvc_ml_pipeline", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ThE-GuY-sHuBhAm/dvc_ml_pipeline.mlflow")

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return metrics and predictions."""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred)
        }

        logger.debug('Model evaluation completed with metrics: %s', metrics)
        return metrics, y_pred

    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def generate_confusion_matrix(y_true, y_pred):
    """
    Generate a confusion matrix figure.
    Returns a matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")

    return fig

def save_model_info(run_id: str, model_path: str, info_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(info_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', info_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("dvc-pipeline-imdb-movies-review")

    with mlflow.start_run() as run:
        try:
            clf = load_model('ml-pipeline-imdb-movies-review/models/model.pkl')
            test_data = load_data('ml-pipeline-imdb-movies-review/data/processed/test_tfidf.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics, y_pred = evaluate_model(clf, X_test, y_test)

            # log params
            if hasattr(clf, "n_estimators"):
                mlflow.log_param("n_estimators", clf.n_estimators)

            # log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            save_metrics(
                metrics,
                'ml-pipeline-imdb-movies-review/reports/metrics.json'
            )

            mlflow.log_artifact(
                "ml-pipeline-imdb-movies-review/reports/metrics.json",
                "metrics"
            )

            # 🔥 MODEL LOGGING (CRITICAL PART)
            print("LOGGING MODEL TO MLFLOW")

            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model"
            )

            print("MODEL LOGGED")

            # save model info for registry
            save_model_info(
                run.info.run_id,
                "model",
                "ml-pipeline-imdb-movies-review/reports/experiment_info.json"
            )

            # confusion matrix
            fig = generate_confusion_matrix(y_test, y_pred)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            raise

if __name__ == '__main__':
    main()





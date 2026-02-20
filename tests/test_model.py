import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. SETUP CREDENTIALS
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        dagshub_url = "https://dagshub.com"
        repo_owner = "ThE-GuY-sHuBhAm"
        repo_name = "dvc_ml_pipeline"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # 2. LOCATE MODEL IN REGISTRY
        cls.model_name = "model"
        cls.stage = "Staging"
        
        client = MlflowClient()
        # Get the latest version in Staging
        latest_versions = client.get_latest_versions(cls.model_name, stages=[cls.stage])
        if not latest_versions:
            raise Exception(f"No model found in {cls.stage} stage to test.")
            
        run_id = latest_versions[0].run_id
        
        # 3. DOWNLOAD & LOAD RAW MODEL
        print(f"Downloading model artifact from Run ID: {run_id}...")
        try:
            # Download the 'model_backup' folder
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="model_backup"
            )
            
            # Look for model.pkl inside the downloaded folder
            model_file_path = os.path.join(local_path, "model.pkl")
            
            # Load raw pickle
            with open(model_file_path, 'rb') as f:
                cls.new_model = pickle.load(f)
                
        except Exception as e:
            raise Exception(f"Failed to load model from registry: {e}")

        # 4. LOAD LOCAL RESOURCES
        # Use raw strings (r'...') to fix path warnings
        try:
            with open(r'ml-pipeline-imdb-movies-review/models/vectorizer.pkl', 'rb') as f:
                cls.vectorizer = pickle.load(f)
            cls.holdout_data = pd.read_csv(r'ml-pipeline-imdb-movies-review/data/processed/test_tfidf.csv')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Local resources not found. Did you run dvc repro? Error: {e}")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input
        input_text = "I Loved this movie! It was fantastic and thrilling."
        input_data = self.vectorizer.transform([input_text])
        
        # Note: The model was trained on numpy arrays, so we pass the array directly
        # to avoid warnings about feature names
        input_array = input_data.toarray()

        # Predict
        prediction = self.new_model.predict(input_array)

        # Verify input features match
        self.assertEqual(input_array.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify output
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        # Extract features and labels
        X_holdout = self.holdout_data.iloc[:, 0:-1].values # Convert to numpy array
        y_holdout = self.holdout_data.iloc[:, -1].values

        # Predict
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Thresholds based on previous model performance (these are just examples, adjust as needed) 
        expected_accuracy = 0.80
        expected_precision = 0.78
        expected_recall = 0.85
        expected_f1 = 0.81

        print(f"\nTest Metrics -> Acc: {accuracy_new:.2f}, Prec: {precision_new:.2f}, Rec: {recall_new:.2f}, F1: {f1_new:.2f}")

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy {accuracy_new} < {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision {precision_new} < {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall {recall_new} < {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 {f1_new} < {expected_f1}')

if __name__ == "__main__":
    unittest.main()
import mlflow
import subprocess
import yaml

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "imdb_sentiment_pipeline"

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_stage(cmd, stage_name):
    print(f"\n--- Running {stage_name} ---")
    subprocess.check_call(cmd, shell=True)

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = load_params()

    with mlflow.start_run(run_name="gb_tfidf_v1"):

        # -------------------------------
        # Data Ingestion parameters
        # -------------------------------
        mlflow.log_param(
            "test_size",
            params["data_ingestion"]["test_size"]
        )

        # -------------------------------
        # Feature engineering parameters
        # -------------------------------
        mlflow.log_param(
            "tfidf_max_features",
            params["feature_engineering"]["max_features"]
        )
        mlflow.log_param(
            "tfidf_ngram_range",
            tuple(params["feature_engineering"]["ngram_range"])
        )

        # -------------------------------
        # Model building parameters
        # -------------------------------
        mlflow.log_param(
            "gb_n_estimators",
            params["model_building"]["n_estimators"]
        )
        mlflow.log_param(
            "gb_learning_rate",
            params["model_building"]["learning_rate"]
        )

        # -------------------------------
        # Run pipeline stages
        # -------------------------------
        run_stage("python data_ingestion.py", "Data Ingestion")
        run_stage("python data_preprocessing.py", "Data Preprocessing")
        run_stage("python feature_engineering.py", "Feature Engineering")
        run_stage("python model_building.py", "Model Building")
        run_stage("python model_evaluation.py", "Model Evaluation")

        # -------------------------------
        # Metadata / tags
        # -------------------------------
        mlflow.set_tag("pipeline", "imdb_sentiment")
        mlflow.set_tag("model", "GradientBoosting + TFIDF")

if __name__ == "__main__":
    main()

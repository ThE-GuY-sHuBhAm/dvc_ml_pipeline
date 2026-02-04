import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # ---------------------------------------------------------
    # 1. SETUP CREDENTIALS
    # ---------------------------------------------------------
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "ThE-GuY-sHuBhAm"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "ThE-GuY-sHuBhAm"
    repo_name = "dvc_ml_pipeline"

    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = MlflowClient()

    # ---------------------------------------------------------
    # 2. SELECT MODEL
    # ---------------------------------------------------------
    model_name = "model"  # <--- FIXED: Must match register_model.py

    # Get the latest version in staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    
    if not staging_versions:
        print(f"⚠️ No models found in 'Staging' for '{model_name}'. Promotion aborted.")
        return

    # Use the most recent Staging version
    latest_staging_version = staging_versions[0].version
    run_id = staging_versions[0].run_id
    print(f"✅ Found Model Version {latest_staging_version} in Staging (Run ID: {run_id})")

    # ---------------------------------------------------------
    # 3. ARCHIVE OLD PRODUCTION MODELS
    # ---------------------------------------------------------
    print("Checking for existing Production models...")
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    for version in prod_versions:
        print(f"Archiving Production Version {version.version}...")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # ---------------------------------------------------------
    # 4. PROMOTE NEW MODEL
    # ---------------------------------------------------------
    print(f"🚀 Promoting Version {latest_staging_version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production"
    )
    
    print(f"SUCCESS: Model Version {latest_staging_version} is now in Production.")

if __name__ == "__main__":
    promote_model()
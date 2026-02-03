import json
import logging
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import os

# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ThE-GuY-sHuBhAm"
repo_name = "dvc_ml_pipeline"

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------------------------------------
# 2. CONSTANTS
# --------------------------------------------------
MODEL_NAME = "model"
MODEL_INFO_PATH = "ml-pipeline-imdb-movies-review/reports/experiment_info.json"

def register_model():
    # Load run metadata
    try:
        with open(MODEL_INFO_PATH, "r") as f:
            model_info = json.load(f)
    except FileNotFoundError:
        logger.error(f"Could not find {MODEL_INFO_PATH}. Did model_evaluation.py run successfully?")
        return

    run_id = model_info["run_id"]
    client = MlflowClient()

    logger.info(f"Checking artifacts for Run ID: {run_id}")

    # --------------------------------------------------
    # 3. SMART ARTIFACT DETECTION
    # --------------------------------------------------
    # We check which folder actually exists in the cloud
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    logger.info(f"Available artifacts: {artifact_paths}")

    # Priority 1: Standard 'model' folder
    if "model" in artifact_paths:
        model_source = f"runs:/{run_id}/model"
        logger.info("Found standard 'model' artifact. Using it.")
    
    # Priority 2: Backup 'model_backup' (The one you have!)
    elif "model_backup" in artifact_paths:
        model_source = f"runs:/{run_id}/model_backup"
        logger.warning("Standard 'model' artifact missing. Falling back to 'model_backup'.")
    
    else:
        logger.error("CRITICAL: No model artifacts found to register.")
        return

    # --------------------------------------------------
    # 4. REGISTRATION (Using Client API)
    # --------------------------------------------------
    # Ensure registered model name exists
    try:
        client.get_registered_model(MODEL_NAME)
        logger.info(f"Registered model name '{MODEL_NAME}' already exists.")
    except Exception:
        client.create_registered_model(MODEL_NAME)
        logger.info(f"Created new registered model name '{MODEL_NAME}'.")

    # Create new version using the source we found
    try:
        # We use create_model_version directly to bypass some strict checks
        version = client.create_model_version(
            name=MODEL_NAME,
            source=model_source,
            run_id=run_id
        )
        
        logger.info(f"Successfully registered model version {version.version} from {model_source}")
        
        # Transition to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version.version,
            stage="Staging"
        )
        logger.info("Transitioned to Staging.")
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")

if __name__ == "__main__":
    register_model()
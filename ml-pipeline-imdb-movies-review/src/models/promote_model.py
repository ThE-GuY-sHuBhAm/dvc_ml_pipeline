import os
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Promotion rules
# -----------------------------
MIN_F1 = 0.81           
IMPROVEMENT_MARGIN = 0 

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

    model_name = "model"

    # ---------------------------------------------------------
    # 2. FETCH CHALLENGER (STAGING MODEL)
    # ---------------------------------------------------------
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model in Staging. Promotion aborted.")
        return

    challenger = staging_versions[0]
    challenger_run = client.get_run(challenger.run_id)
    challenger_f1 = challenger_run.data.metrics.get("f1_score")

    if challenger_f1 is None:
        raise ValueError("Challenger model has no F1 metric logged in MLflow.")

    print(f"Challenger (Staging) F1 = {challenger_f1:.4f}")

    # ---------------------------------------------------------
    # 3. FETCH CHAMPION (PRODUCTION MODEL)
    # ---------------------------------------------------------
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    if prod_versions:
        champion = prod_versions[0]
        champion_run = client.get_run(champion.run_id)
        champion_f1 = champion_run.data.metrics.get("f1_score")

        if champion_f1 is None:
            raise ValueError("Production model has no F1 metric logged in MLflow.")

        print(f"Champion (Production) F1 = {champion_f1:.4f}")
    else:
        champion_f1 = None
        print("No Production model found (first deployment).")

    # ---------------------------------------------------------
    # 4. PROMOTION DECISION LOGIC
    # ---------------------------------------------------------
    promote = False

    if challenger_f1 < MIN_F1:
        print(f"Challenger failed minimum quality gate (F1 < {MIN_F1})")

    elif champion_f1 is None:
        print("No champion exists — promoting challenger.")
        promote = True

    elif challenger_f1 > champion_f1 + IMPROVEMENT_MARGIN:
        print("Challenger beats champion — promoting.")
        promote = True

    else:
        print("Challenger did NOT outperform champion — promotion blocked.")

    # ---------------------------------------------------------
    # 5. EXECUTE PROMOTION
    # ---------------------------------------------------------
    if promote:
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

        client.transition_model_version_stage(
            name=model_name,
            version=challenger.version,
            stage="Production"
        )

        print(f"SUCCESS: Model v{challenger.version} promoted to Production.")

    else:
        print("Promotion skipped.")

if __name__ == "__main__":
    promote_model()

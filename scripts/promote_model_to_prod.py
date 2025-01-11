import mlflow
import dagshub
import json
from mlflow import MlflowClient
import os

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Ubaidmalik9567"
repo_name = "delivery-time-prediction"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# get model name
model_name = load_model_information("reports/run_information.json")["model_name"]
stage = "Staging"

# get the latest version from staging stage
client = MlflowClient()

# get the latest version of model in staging
latest_versions = client.get_latest_versions(name=model_name,stages=[stage])

latest_model_version_staging = latest_versions[0].version

# promotion stage
promotion_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)
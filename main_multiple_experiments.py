import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    csv_url = "https://raw.githubusercontent.com/myuser114/wine-dataset/refs/heads/main/winequality.csv"

    # os.mkdir("data/", )

    try:
        data = pd.read_csv(csv_url, sep=';')

        data.to_csv("data/red-wine-quality.csv", index=False)
    except Exception as e:
        logger.exception("Unable to download", e)

    X = data.drop(["quality"], axis=1)
    y = data['quality']

    X.to_csv("data/train.csv", index=False)
    y.to_csv("data/test.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'ElasticNet': ElasticNet(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42)
    }


    for model_name, model in models.items():
        print(f"Model Experiment {model_name}")

        mlflow.set_experiment(experiment_name=f"{model_name}_experiment")

        experiment = mlflow.get_experiment_by_name(f"{model_name}_experiment")

        print(f"Name: f{experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Created Time: {experiment.creation_time}")

        with mlflow.start_run(run_name="run") as run:
            mlflow.set_tag("model", model_name)

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            rmse, mae, r2 = eval_metrics(y_test, pred)

            print(f"Model: {model_name}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2: {r2}")

            mlflow.log_params(model.get_params())

            mlflow.log_metrics(
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            )

            mlflow.sklearn.log_model(model, f"model_{model_name}")

            mlflow.log_artifacts('data/', artifact_path="datas")

            artifact_uri = mlflow.get_artifact_uri(artifact_path="datas")

            active_run = mlflow.active_run()
            print(f"Active Run: {active_run.info.status}")
            print(f"Active Run Name: {active_run.info.run_name}")

            print(f"Artifact URI: {artifact_uri}")
            print(f"Run ID: {active_run.info.run_id}")

            print(mlflow.last_active_run().info.status)

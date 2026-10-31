import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os




logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get parameters from command line
parser = argparse.ArgumentParser("Teste")
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

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

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is: {}".format(mlflow.get_tracking_uri()))

    exp = mlflow.set_experiment(experiment_name="experiment_5")
    get_exp = mlflow.get_experiment(exp.experiment_id)

    print("Name: {}".format(get_exp.name))
    print("Experiment ID: {}".format(get_exp.experiment_id))
    print("Artifact Location: {}".format(get_exp.artifact_location))
    print("Tags: {}".format(get_exp.tags))
    print("Lifecycle Stage: {}".format(get_exp.lifecycle_stage))
    print("Created Time: {}".format(get_exp.creation_time))

    with mlflow.start_run(run_name="run1") as run:
    # with mlflow.start_run(run_id="6d063490f03b4e288996de06822159cb") as run:
    # mlflow.start_run(run_id="6d063490f03b4e288996de06822159cb")

        mlflow.set_tags({"version": "1.0", "model": "regressor"})

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        signature = infer_signature(X_train.iloc[[0]], y_pred)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("ElasticNet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_params(
            {
                "alpha": alpha,
                "l1_ratio": l1_ratio
            }
        )

        mlflow.log_metrics(
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        )

        mlflow.sklearn.log_model(lr, "demo_model2", code_paths=["main.py"], signature=signature, input_example=X_train.iloc[[0]])

        mlflow.log_artifacts('data/', artifact_path="datas")

        artifact_uri = mlflow.get_artifact_uri(artifact_path="datas")

        active_run = mlflow.active_run()
        print(f"Active Run: {active_run.info.status}")
        print(f"Active Run Name: {active_run.info.run_name}")

        print(f"Artifact URI: {artifact_uri}")
        print(f"Run ID: {active_run.info.run_id}")

        print(mlflow.last_active_run().info.status)






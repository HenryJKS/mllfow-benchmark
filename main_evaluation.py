import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import make_metric
import matplotlib.pyplot as plt
import os
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get parameters from command line
parser = argparse.ArgumentParser("Teste")
parser.add_argument("--alpha", type=float, required=False, default=0.001)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.001)
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

    exp = mlflow.set_experiment(experiment_name="experiment_evaluation")
    get_exp = mlflow.get_experiment(exp.experiment_id)

    print("Name: {}".format(get_exp.name))
    print("Experiment ID: {}".format(get_exp.experiment_id))
    print("Artifact Location: {}".format(get_exp.artifact_location))
    print("Tags: {}".format(get_exp.tags))
    print("Lifecycle Stage: {}".format(get_exp.lifecycle_stage))
    print("Created Time: {}".format(get_exp.creation_time))

    with mlflow.start_run() as run:
    # with mlflow.start_run(run_id="6d063490f03b4e288996de06822159cb") as run:
    # mlflow.start_run(run_id="6d063490f03b4e288996de06822159cb")

        mlflow.set_tags({"version": "1.0", "model": "regressor"})

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        signature = infer_signature(X_train.iloc[[0]], y_pred)

        mlflow.sklearn.log_model(lr, "demo_model2", code_paths=["main.py"], signature=signature, input_example=X_train.iloc[[0]])

        eval_df = X_test.copy()
        eval_df["quality"] = y_test

        def squared_diff_plus_one(eval_df, _builtin_metrics):
            return np.sum(np.abs(eval_df['prediction'] - eval_df['target'] + 1) ** 2)

        def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
            return builtin_metrics["sum_on_target"] / 2

        squared_diff_plus_one_metric = make_metric(eval_fn=squared_diff_plus_one, greater_is_better=False, name="squared diff plus one")
        sum_on_target_divided_by_two_metrics = make_metric(eval_fn=sum_on_target_divided_by_two, greater_is_better=True, name="sum on target divided by two")

        def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
            plt.scatter(eval_df["prediction"], eval_df["target"])
            plt.xlabel("Targets")
            plt.ylabel("Predictions")
            plt.title("Target vs Prediction")
            plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
            plt.savefig(plot_path)

            return {"example_scatter_plot": plot_path}

        get_artifact_uri = mlflow.get_artifact_uri("demo_model2")

        mlflow.evaluate(
            get_artifact_uri,
            eval_df,
            targets="quality",
            model_type="regressor",
            evaluators=["default"],
            custom_metrics=[squared_diff_plus_one_metric, sum_on_target_divided_by_two_metrics],
            custom_artifacts=[prediction_target_scatter]
        )

        print("ElasticNet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))


        mlflow.log_params(lr.get_params())

        active_run = mlflow.active_run()
        print(f"Active Run: {active_run.info.status}")
        print(f"Active Run Name: {active_run.info.run_name}")

        print(mlflow.last_active_run().info.status)






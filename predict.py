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
import os
import sklearn
import joblib
import cloudpickle


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

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

    load_model = mlflow.pyfunc.load_model(model_uri="runs:/01765ba49ced433d832a655985dd05d0/sklearn_mlflow_pyfunc")

    X_train_pred = X_train.iloc[[0]]
    pred = load_model.predict(X_train_pred)
    print(pred)

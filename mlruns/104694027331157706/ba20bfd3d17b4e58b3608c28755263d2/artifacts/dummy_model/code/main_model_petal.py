import mlflow
from numpy.ma.extras import average
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from mlflow.models import MetricThreshold, make_metric, infer_signature
import os
import matplotlib.pyplot as plt
# Leitura dos dados
df = pd.read_csv("raw_data/iris.csv", sep=",")

# Artifacts
df.to_csv('data/raw_data', index=False)

print(df['variety'].unique())

le = LabelEncoder()

df['label'] = le.fit_transform(df['variety'])

df = df.drop('variety', axis=1)

X = df.drop('label', axis=1)
y = df['label']

print(np.unique(df['label'], return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Artifacts
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

experiment = mlflow.set_experiment(experiment_name="iris-classification")
print(f"Experiment ID: {experiment.experiment_id}")

with mlflow.start_run():
    mlflow.set_tags({
        'model': "Logistic Regression",
        "dataset": "iris",
        "version": "1.0"
    })

    lr = LogisticRegression(random_state=42)

    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    signature = infer_signature(X_train.iloc[[0]], pred)

    mlflow.log_artifacts("data/", artifact_path="data_artifacts")
    mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=X_train.iloc[[0]], code_paths=["main_model_petal.py"])

    dummy_model = DummyClassifier(random_state=42)
    dummy_model.fit(X_train, y_train)
    dummy_pred = dummy_model.predict(X_test)

    mlflow.sklearn.log_model(dummy_model, "dummy_model", signature=signature, input_example=X_train.iloc[[0]], code_paths=["main_model_petal.py"])

    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    precision = precision_score(y_test, pred, average='weighted')
    accuracy = accuracy_score(y_test, pred)

    print(f"Recall: {round(recall * 100, 2)}")
    print(f"F1: {round(f1 * 100, 2)}")
    print(f"Precision: {round(precision * 100, 2)}")
    print(f"Accuracy: {round(accuracy * 100, 2)}")

    mlflow.log_params(lr.get_params())

    def accuracy_score(eval_df, _builtin_metrics):
        return accuracy_score(eval_df["label"], eval_df["prediction"])

    def precision_score_weighted(eval_df, _builtin_metrics):
        return precision_score(eval_df["label"], eval_df["prediction"], average='weighted')

    def f1_score_weighted(eval_df, _builtin_metrics):
        return f1_score(eval_df["label"], eval_df["prediction"], average='weighted')

    def recall_score_weighted(eval_df, _builtin_metrics):
        return recall_score(eval_df["label"], eval_df["prediction"], average='weighted')

    accuracy_score_metric = make_metric(eval_fn=accuracy_score, greater_is_better=True, name="accuracy")
    precision_score_weighted_metrics = make_metric(eval_fn=precision_score_weighted, greater_is_better=True, name="precision_weighted")
    f1_score_weighted_metrics = make_metric(eval_fn=f1_score_weighted, greater_is_better=True, name="f1_weighted")
    recall_score_weighted_metrics = make_metric(eval_fn=recall_score_weighted, greater_is_better=True, name="recall_weighted")

    def plot_confusion_matrix(eval_df, _builtin_metrics, artifacts_dir):
        cm = confusion_matrix(eval_df["label"], eval_df["prediction"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(plot_path)
        return {"confusion_matrix": plot_path}

    eval_df = X_test.copy()
    eval_df["label"] = y_test

    model_artifact_uri = mlflow.get_artifact_uri("model")
    baseline_artifact_uri = mlflow.get_artifact_uri("dummy_model")

    thresholds = {
        "accuracy": MetricThreshold(
            threshold=0.8,
            greater_is_better=True
        )
    }

    mlflow.evaluate(
        model_artifact_uri,
        eval_df,
        model_type='classifier',
        evaluators=["default"],
        custom_metrics=[accuracy_score_metric, precision_score_weighted_metrics, f1_score_weighted_metrics, recall_score_weighted_metrics],
        custom_artifacts=[plot_confusion_matrix],
        validation_thresholds=thresholds,
        baseline_model=baseline_artifact_uri
    )



last_active_run = mlflow.last_active_run()
print(last_active_run.info.run_name)
print(last_active_run.info.run_id)
print(last_active_run.info.status)
print(mlflow.get_artifact_uri(artifact_path="data_artifacts"))







from pathlib import Path
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("dataset", type=str)

    args = parser.parse_args()

    curr_dir = Path(__file__).parent.absolute()
    dataset_path = curr_dir / args.dataset
    df = pd.read_csv(dataset_path)

    X = df.drop('covid_result', axis=1)
    y = df['covid_result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow_path = curr_dir / "mlruns"
        if os.name == 'nt':  # Windows
            tracking_uri = mlflow_path.as_uri()
        else:  # Linux (GitHub Actions)
            tracking_uri = f"file://{mlflow_path.as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run(nested=True):
        rf = RandomForestClassifier(random_state=42, max_depth=args.max_depth, n_estimators=args.n_estimators)

        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        metrics = {
            "test_accuracy": acc,
            "test_precision": precision_score(y_test, predictions, average='weighted'),
            "test_recall": recall_score(y_test, predictions, average='weighted'),
            "test_f1_score": f1_score(y_test, predictions, average='weighted')
        }
        mlflow.log_metrics(metrics)

        if "GITHUB_OUTPUT" in os.environ:
            run = mlflow.active_run()
            run_id = run.info.run_id
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"run_id={run_id}\n")

from pathlib import Path
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

curr_dir = Path(__file__).parent.absolute()
dataset_path = curr_dir / "covid19-patient-symptoms-diagnosis_preprocessing.csv"
df = pd.read_csv(dataset_path)

X = df.drop('covid_result', axis=1)
y = df['covid_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow_path = curr_dir / "mlruns"

if os.name == 'nt':  # Windows
    tracking_uri = mlflow_path.as_uri()
else:  # Linux (GitHub Actions)
    tracking_uri = f"file://{mlflow_path.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Covid19 Patient Symptoms Diagnosis Github")
mlflow.sklearn.autolog(log_input_examples=True)

with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42, max_depth=12, n_estimators=32)

    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": acc,
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1_score": f1_score(y_test, predictions, average='weighted')
    }
    mlflow.log_metrics(metrics)
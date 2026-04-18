from pathlib import Path
import pandas as pd
import joblib
import yaml
from loguru import logger
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from mlflow import tracking
from youtube_comment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

import os
import tempfile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

def evaluate(test_features_path: Path, model_dir: Path):
    params = load_params()
    logger.info("Loading test data and model for evaluation...")
    test_df = pd.read_csv(test_features_path)
    clf = joblib.load(model_dir / "model.pkl")
    tfidf = joblib.load(model_dir / "tfidf.pkl")
    
    processed_col = params['columns']['text_processed']
    target_col = params['columns']['target']
    numeric_cols = params['columns']['numeric_features']

    # Preprocess test data
    X_tfidf = tfidf.transform(test_df[processed_col])
    X_numeric = test_df[numeric_cols].values
    X_combined = hstack([X_tfidf, X_numeric])
    
    y_test = test_df[target_col]
    y_pred = clf.predict(X_combined)
    
    # ── Detailed Metrics (Matching Notebook Template) ────────────────────────
    report_dict: dict = classification_report(y_test, y_pred, output_dict=True) # type: ignore
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": report_dict['weighted avg']['f1-score'],
        "f1_macro": report_dict['macro avg']['f1-score'],
        "precision_macro": report_dict['macro avg']['precision'],
        "recall_macro": report_dict['macro avg']['recall']
    }
    
    # Add class-specific metrics
    for label in report_dict:
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f"f1_class_{label}"] = report_dict[label]['f1-score']
            metrics[f"precision_class_{label}"] = report_dict[label]['precision']
            metrics[f"recall_class_{label}"] = report_dict[label]['recall']
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{name}: {value:.4f}")
    
    with mlflow.start_run(nested=True, run_name="Evaluation") as run:
        mlflow.log_metrics(metrics)
        
        # ── Industrial Practice: Save metrics for DVC ────────────────────────
        import json
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # ── Industrial Practice: Registry Management (Aliases) ───────────────
        # In MLflow 2.9+, 'Stages' are deprecated. We now use 'Aliases'.
        # New models are tagged as 'challenger' for validation.
        try:
            client = tracking.MlflowClient()
            model_name = params['base']['project']
            
            # Find the latest version of this model
            latest_version = client.get_latest_versions(model_name)[0].version
            
            # Set alias to 'challenger'
            client.set_registered_model_alias(
                name=model_name,
                alias="challenger",
                version=latest_version
            )
            logger.success(f"Model {model_name} v{latest_version} tagged as @challenger")
        except Exception as e:
            logger.warning(f"Could not set model alias: {e}")

        # Log Classification Report as Text
        report_str: str = classification_report(y_test, y_pred) # type: ignore
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = os.path.join(tmp_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report_str)
            mlflow.log_artifact(report_path)

            # Confusion Matrix Plot
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_path = FIGURES_DIR / "confusion_matrix.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(str(plot_path))
        
    logger.success("Evaluation complete.")

if __name__ == "__main__":
    # In production, we evaluate using the test features generated in the features stage
    evaluate(
        test_features_path=PROCESSED_DATA_DIR / "test_features.csv",
        model_dir=MODELS_DIR
    )

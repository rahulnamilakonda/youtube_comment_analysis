import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib
from loguru import logger
import mlflow
from mlflow import sklearn
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack
from youtube_comment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train(features_path: Path, model_dir: Path):
    params = load_params()
    
    # MLflow Setup (Dagshub placeholders)
    # These can be set via environment variables
    # os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/user/repo.mlflow'
    # os.environ['MLFLOW_TRACKING_USERNAME'] = 'your_username'
    # os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your_token'
    
    mlflow.set_experiment(params['base']['project'])
    
    with mlflow.start_run():
        logger.info(f"Loading training features from {features_path}")
        train_df = pd.read_csv(features_path)
        
        # 1. TF-IDF Vectorization
        logger.info("Fitting TF-IDF...")
        tfidf_params = params['features']['tfidf']
        processed_col = params['columns']['text_processed']
        target_col = params['columns']['target']
        
        tfidf = TfidfVectorizer(
            max_features=tfidf_params['max_features'],
            ngram_range=tuple(tfidf_params['ngram_range'])
        )
        X_tfidf = tfidf.fit_transform(train_df[processed_col])
        
        # 2. Combine with numeric features
        numeric_cols = params['columns']['numeric_features']
        X_numeric = train_df[numeric_cols].values
        X_combined = hstack([X_tfidf, X_numeric]).toarray()
        
        # Industrial Practice: Convert to DataFrame with string column names
        # This ensures LightGBM tracks feature names and MLflow signature is clean.
        X_combined_df = pd.DataFrame(X_combined)
        X_combined_df.columns = [str(i) for i in range(X_combined_df.shape[1])]
        
        y = train_df[target_col]
        
        # 3. Train Model
        logger.info("Training LightGBM model...")
        model_params = params['train']['params']
        clf = LGBMClassifier(**model_params)
        clf.fit(X_combined_df, y)

        # 4. Detailed Metrics Logging (Train & Test)
        def get_metrics(y_true, y_pred, prefix):
            report: dict = classification_report(y_true, y_pred, output_dict=True) # type: ignore
            metrics = {
                f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
                f"{prefix}_f1_weighted": report['weighted avg']['f1-score'],
                f"{prefix}_f1_macro": report['macro avg']['f1-score'],
                f"{prefix}_precision_macro": report['macro avg']['precision'],
                f"{prefix}_recall_macro": report['macro avg']['recall']
            }
            for label in report:
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    metrics[f"{prefix}_f1_class_{label}"] = report[label]['f1-score']
                    metrics[f"train_precision_class_{label}"] = report[label]['precision']
                    metrics[f"train_recall_class_{label}"] = report[label]['recall']
            return metrics

        # Log Train Metrics
        logger.info("Logging training metrics...")
        y_train_pred = clf.predict(X_combined_df)
        train_metrics = get_metrics(y, y_train_pred, "train")
        mlflow.log_metrics(train_metrics)
        
        # 5. Logging Parameters
        # Industrial Practice: Log both high-level metadata and specific model params
        metadata = {
            "test_size": params['preprocessing']['train_test_split'],
            "stratify": True,
            "representation": f"TFIDF_{params['features']['tfidf']['max_features']}",
            "scaler": f"{params['features']['scaling'].capitalize()}Scaler",
            "model_name": "LightGBM_Optimized",
            "resampler": "none",
            "leak_proof": True
        }
        mlflow.log_params(metadata)
        mlflow.log_params(clf.get_params()) # Logs all LightGBM params (including defaults)
        mlflow.log_params(tfidf_params)
        
        # 5. Save artifacts
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_dir / "model.pkl")
        joblib.dump(tfidf, model_dir / "tfidf.pkl")
        
        # Log model to MLflow (using sklearn flavor for LGBMClassifier)
        # Industrial Practice: Adding Model Signature (Schema) for production safety
        from mlflow.models import infer_signature
        
        # We ensure X_combined_df has string column names and is a DataFrame
        # for clean signature inference.
        signature = infer_signature(X_combined_df[:5], y_train_pred[:5]) # type: ignore
        
        # We use input_example to make the schema visible in the MLflow UI
        sklearn.log_model(     
            sk_model=clf, 
            artifact_path="model",
            signature=signature,
            input_example=X_combined_df[:5],
            registered_model_name=params['base']['project']
        )
        
        logger.success(f"Model and vectorizer saved to path: {model_dir}")

if __name__ == "__main__":
    train(
        features_path=PROCESSED_DATA_DIR / "train_features.csv",
        model_dir=MODELS_DIR
    )

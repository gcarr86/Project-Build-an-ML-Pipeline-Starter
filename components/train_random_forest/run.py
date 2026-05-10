import argparse
import json
import pandas as pd
import wandb
import mlflow.sklearn
import os
import shutil

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_inference_pipeline(rf_config):

    numerical_features = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365"
    ]

    categorical_features = [
        "neighbourhood_group",
        "room_type"
    ]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    non_ordinal_categorical_preproc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", non_ordinal_categorical_preproc, categorical_features)
        ]
    )

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", RandomForestRegressor(**rf_config))
        ]
    )

    return sk_pipe


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(vars(args))

    # Load train data
    train_artifact = run.use_artifact(args.train_artifact)
    train_df = pd.read_csv(train_artifact.file())

    # Load validation data
    val_artifact = run.use_artifact(args.val_artifact)
    val_df = pd.read_csv(val_artifact.file())

    # Load model config
    with open(args.model_config) as fp:
        model_config = json.load(fp)

    rf_config = model_config["random_forest"].copy()
    rf_config["random_state"] = args.random_seed

    # Split data
    x_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]

    x_val = val_df.drop(columns=["price"])
    y_val = val_df["price"]

    # Train model
    sk_pipe = get_inference_pipeline(rf_config)
    sk_pipe.fit(x_train, y_train)

    # Predict
    preds = sk_pipe.predict(x_val)

    # Evaluate
    mae = mean_absolute_error(y_val, preds)
    run.summary["MAE"] = mae

    # Export model
    export_path = "model_export"

    if os.path.isdir(export_path):
        shutil.rmtree(export_path, ignore_errors=True)

    mlflow.sklearn.save_model(
        sk_model=sk_pipe,
        path=export_path
    )

    # Log artifact (IMPORTANT FIXED SECTION)
    artifact = wandb.Artifact(
        name="model_export",
        type="model_export",
        description="Random forest pipeline export"
    )

    artifact.add_dir(export_path)
    run.log_artifact(artifact)

    # ✅ CRITICAL FIX: always finish run
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_artifact", type=str, required=True)
    parser.add_argument("--val_artifact", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--random_seed", type=int, required=True)

    args = parser.parse_args()
    go(args)
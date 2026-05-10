import argparse
import pandas as pd
import wandb
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error


def go(args):

    run = wandb.init(job_type="evaluate")

    # Load test artifact
    test_artifact = run.use_artifact(args.test_artifact)
    test_path = test_artifact.file()

    test_df = pd.read_csv(test_path)

    # Load model artifact
    model_artifact = run.use_artifact(args.model_export)

    model_path = model_artifact.download()

    model = mlflow.sklearn.load_model(model_path)

    # Split features/target
    x_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Predict
    preds = model.predict(x_test)

    # Evaluate
    mae = mean_absolute_error(y_test, preds)

    print(f"Test MAE: {mae}")

    # Log metric
    run.summary["test_mae"] = mae

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_export",
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_artifact",
        type=str,
        required=True
    )

    args = parser.parse_args()

    go(args)
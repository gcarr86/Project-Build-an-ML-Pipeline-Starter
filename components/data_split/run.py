import argparse
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(vars(args))

    # Load cleaned data
    artifact = run.use_artifact(args.input_artifact)
    artifact_dir = artifact.download()
    df = pd.read_csv(f"{artifact_dir}/clean_sample.csv")

    # Stratification logic
    stratify_col = df[args.stratify_by] if args.stratify_by != "none" else None

    # Train/val/test split
    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col
    )

    stratify_col_train_val = (
        train_val[args.stratify_by] if args.stratify_by != "none" else None
    )

    train, val = train_test_split(
        train_val,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify_col_train_val
    )

    # Save outputs
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    test.to_csv("test.csv", index=False)

    # Log artifacts
    for fname, atype in [
        ("train.csv", "train_data"),
        ("val.csv", "val_data"),
        ("test.csv", "test_data"),
    ]:
        artifact = wandb.Artifact(
            fname,
            type=atype,
            description=f"{atype} split"
        )
        artifact.add_file(fname)
        run.log_artifact(artifact)

    run.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)

    args = parser.parse_args()

    go(args)
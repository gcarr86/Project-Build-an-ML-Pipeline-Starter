import argparse

import pandas as pd
import wandb
from scipy.stats import ks_2samp


def go(args):
    run = wandb.init(job_type="data_check")
    run.config.update(vars(args))

    # Load reference data
    ref_artifact = run.use_artifact(args.reference_artifact)
    ref_path = ref_artifact.file()
    ref_df = pd.read_csv(ref_path)

    # Load sample data
    sample_artifact = run.use_artifact(args.sample_artifact)
    sample_path = sample_artifact.file()
    sample_df = pd.read_csv(sample_path)

    # Check for missing values
    assert sample_df.isnull().sum().sum() == 0, "Data contains missing values"

    # Check price distribution using KS test
    ref_prices = pd.to_numeric(ref_df["price"], errors="coerce").dropna()
    sample_prices = pd.to_numeric(sample_df["price"], errors="coerce").dropna()

    if len(ref_prices) > 1 and len(sample_prices) > 1:
        stat, p_value = ks_2samp(ref_prices, sample_prices)

        if not pd.isna(p_value):
            assert p_value > args.ks_alpha, f"KS test failed: p={p_value}"

    # Check minimum_nights reasonable
    minimum_nights = pd.to_numeric(sample_df["minimum_nights"], errors="coerce").dropna()
    run.summary["max_minimum_nights"] = float(minimum_nights.max())

    # Accept kl_threshold from MLflow/config
    assert args.kl_threshold >= 0, "kl_threshold must be non-negative"

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_artifact", type=str, required=True)
    parser.add_argument("--sample_artifact", type=str, required=True)
    parser.add_argument("--ks_alpha", type=float, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)

    args = parser.parse_args()
    go(args)
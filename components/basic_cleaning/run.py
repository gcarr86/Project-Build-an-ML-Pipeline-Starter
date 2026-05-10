#!/usr/bin/env python
"""
Basic cleaning script for the Udacity ML Pipeline project.
Downloads a raw CSV artifact from W&B, cleans it, and uploads the cleaned CSV back to W&B.
"""

import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Download raw data from W&B, apply basic cleaning, and upload the cleaned data.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing input artifact, output artifact,
        output type, output description, min price, and max price.
    """

    run = wandb.init(job_type="basic_cleaning")

    logger.info("Downloading input artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info(f"Loading data from {artifact_path}")
    df = pd.read_csv(artifact_path, sep=",", engine="python")

    logger.info("Cleaning and filtering price outliers")

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    logger.info(f"Rows after price filtering: {len(df)}")

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Filtering minimum_nights outliers")

    df["minimum_nights"] = pd.to_numeric(
        df["minimum_nights"],
        errors="coerce"
    )

    df = df[df["minimum_nights"] <= 365].copy()

    logger.info("Renaming columns if needed")
    df = df.rename(
        columns={
            "neighbourhood_group_cleansed": "neighbourhood_group",
            "neighbourhood_cleansed": "neighbourhood",
        }
    )

    required_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    logger.info("Keeping required columns")
    df = df[required_columns].copy()

    logger.info("Filtering to NYC geographic boundaries")

    df = df[
        df["longitude"].between(-74.25, -73.50)
        & df["latitude"].between(40.5, 41.2)
    ].copy()
    
    logger.info("Filling remaining missing values")

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["last_review"] = df["last_review"].fillna(pd.Timestamp("1970-01-01"))

    df = df.dropna().reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, args.output_artifact)

        logger.info(f"Saving cleaned data to {output_path}")
        df.to_csv(output_path, index=False)

        logger.info("Uploading cleaned artifact to W&B")
        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(output_path)

        run.log_artifact(artifact)

    run.finish()
    logger.info("Cleaning complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input W&B artifact, for example sample.csv:latest",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price",
        required=True,
    )

    args = parser.parse_args()
    go(args)
#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import os
import wandb


def go(args):
    run = wandb.init(job_type="download")
    run.config.update(vars(args))

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(args.sample)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("sample", type=str)
    parser.add_argument("artifact_name", type=str)
    parser.add_argument("artifact_type", type=str)
    parser.add_argument("artifact_description", type=str)

    args = parser.parse_args()
    go(args)
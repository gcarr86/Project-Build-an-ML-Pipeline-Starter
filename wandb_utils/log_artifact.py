import wandb

def log_artifact(artifact_name, artifact_type, artifact_description, file_path, run):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description
    )
    artifact.add_file(file_path)
    run.log_artifact(artifact)

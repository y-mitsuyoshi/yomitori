"""SageMaker Local Mode training script.

Run with: python -m scripts.local_train

Prerequisites:
  - Docker installed and Docker socket accessible
  - Synthetic data generated at data/synthetic/driver_license/
"""

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DUMMY_ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-Dummy"


def main() -> None:
    """Run a SageMaker Local Mode training job."""
    import sagemaker
    from sagemaker.pytorch import PyTorch

    project_root = Path(__file__).resolve().parent.parent

    hyperparameters = {
        "data_dir": "/opt/ml/input/data/training",
        "output_dir": "/opt/ml/model",
        "epochs": "3",
        "batch_size": "4",
        "fp16": "True",
    }

    estimator = PyTorch(
        entry_point="training/train_trocr.py",
        source_dir=str(project_root),
        role=_DUMMY_ROLE,
        instance_count=1,
        instance_type="local_gpu",
        image_uri="yomitori:train",
        hyperparameters=hyperparameters,
    )

    data_path = f"file://{project_root / 'data' / 'synthetic' / 'driver_license'}"
    estimator.fit({"training": data_path})
    logger.info("Training complete")


if __name__ == "__main__":
    main()
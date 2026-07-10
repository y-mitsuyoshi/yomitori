"""SageMaker Local Mode training script.

Run with: python -m sagemaker.local_train

Prerequisites:
  - Docker installed and Docker socket accessible
  - Synthetic data generated at data/synthetic/driver_license/
"""

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DUMMY_ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-Dummy"


def main() -> None:
    """Launch SageMaker Local Mode training job.

    Uses the local Docker environment to run a training job that mirrors
    the SageMaker cloud training flow. The training container (Dockerfile.train)
    is built and launched by the SageMaker SDK.
    """
    from sagemaker.local import LocalSession
    from sagemaker.pytorch import PyTorch

    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    project_root = Path(__file__).resolve().parent.parent
    data_path = str(project_root / "data" / "synthetic" / "driver_license")

    if not Path(data_path).exists():
        logger.error("Training data not found at %s. Run generate_synthetic_data first.", data_path)
        raise FileNotFoundError(f"Training data not found: {data_path}")

    estimator = PyTorch(
        entry_point="train_trocr.py",
        source_dir=str(project_root),
        role=_DUMMY_ROLE,
        framework_version="2.6",
        py_version="py310",
        instance_count=1,
        instance_type="local_gpu",
        hyperparameters={
            "epochs": 5,
            "batch_size": 8,
            "data_dir": "/opt/ml/input/data/training",
        },
    )
    estimator.fit({"training": f"file://{data_path}"})
    logger.info("Local training job complete. Model at: %s", estimator.model_data)
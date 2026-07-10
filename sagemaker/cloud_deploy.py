"""Cloud deployment script for SageMaker.

Run with: python -m sagemaker.cloud_deploy
"""

import os
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Deploy the model to a SageMaker cloud endpoint."""
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker cloud deploy")
    parser.add_argument("--model_data", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN")
    parser.add_argument(
        "--instance_type", default="ml.g5.xlarge", help="Endpoint instance type"
    )
    args = parser.parse_args()

    import sagemaker
    from sagemaker.pytorch import PyTorchModel

    role = args.role or sagemaker.get_execution_role()

    project_root = Path(__file__).resolve().parent.parent

    model = PyTorchModel(
        model_data=args.model_data,
        role=role,
        entry_point="sagemaker/inference_entry_point.py",
        source_dir=str(project_root),
        framework_version="2.6",
        py_version="py310",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
    )
    logger.info("Endpoint deployed: %s", predictor.endpoint_name)
    print(f"Endpoint name: {predictor.endpoint_name}")
    print("Test with:")
    print('  predictor.predict({"image": "<base64>"})')


if __name__ == "__main__":
    main()
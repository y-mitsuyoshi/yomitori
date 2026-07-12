"""Cloud deployment script for SageMaker.

Run with: python -m scripts.cloud_deploy --model_data s3://... --ecr_image <uri> --role <arn>

Prerequisites:
  - AWS credentials configured (aws configure)
  - model.tar.gz uploaded to S3
  - Docker image pushed to ECR
"""

import argparse
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Deploy the model to a SageMaker cloud endpoint.

    Uses a custom Docker image (yomitori:infer pushed to ECR) that contains
    CUDA 12.8, PyTorch, docTR, and TrOCR. The SageMaker PyTorch pre-built
    images don't include these dependencies.
    """
    parser = argparse.ArgumentParser(description="SageMaker cloud deploy")
    parser.add_argument("--model_data", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--ecr_image", required=True, help="ECR image URI (e.g. xxx.dkr.ecr.region.amazonaws.com/yomitori:infer)")
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
        image_uri=args.ecr_image,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
    )
    logger.info("Endpoint deployed: %s", predictor.endpoint_name)
    print(f"\nEndpoint name: {predictor.endpoint_name}")
    print("\nTest with:")
    print("  # Raw image bytes (recommended)")
    print("  predictor.predict(open('image.jpg', 'rb').read(),")
    print("      initial_args={'ContentType': 'image/jpeg'})")
    print("  # Or JSON with base64")
    print("  import json, base64")
    print("  b64 = base64.b64encode(open('image.jpg', 'rb').read()).decode()")
    print("  predictor.predict(json.dumps({'image': b64}),")
    print("      initial_args={'ContentType': 'application/json'})")


if __name__ == "__main__":
    main()
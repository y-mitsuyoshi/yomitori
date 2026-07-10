"""SageMaker Local Mode deployment and test script.

Run with: python -m sagemaker.local_deploy [--sample <path>]

Prerequisites:
  - local_train.py has been run (model artifacts exist)
  - Docker installed and Docker socket accessible
"""

import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Deploy a local SageMaker endpoint and run a test inference.

    This creates a local Docker-based endpoint mirroring the SageMaker cloud
    inference flow. The inference container (Dockerfile.infer) is built and
    launched by the SageMaker SDK.
    """
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker Local deploy & test")
    parser.add_argument(
        "--sample",
        default="data/samples/sample_license.jpg",
        help="Sample image path",
    )
    args = parser.parse_args()

    from sagemaker.local import LocalSession
    from sagemaker.pytorch import PyTorchModel

    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    project_root = Path(__file__).resolve().parent.parent

    # model.tar.gz を探す
    model_tar = project_root / "model.tar.gz"
    if model_tar.exists():
        model_data = f"file://{model_tar}"
    else:
        logger.error(
            "model.tar.gz not found. Create it first:\n"
            "  docker compose run --rm dev bash -c "
            "\"cd /opt/ml/model && tar czf /opt/ml/code/model.tar.gz .\""
        )
        raise FileNotFoundError("model.tar.gz not found")

    model = PyTorchModel(
        model_data=model_data,
        role="arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-Dummy",
        entry_point="sagemaker/inference_entry_point.py",
        source_dir=str(project_root),
        framework_version="2.6",
        py_version="py310",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
    )

    sample_path = Path(args.sample)
    if sample_path.exists():
        # 生画像バイナリを送信（base64エンコード不要）
        with sample_path.open("rb") as f:
            raw = f.read()
        result = predictor.predict(
            raw,
            initial_args={"ContentType": "image/jpeg"},
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.warning("Sample image not found: %s", sample_path)
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        print("Test with:")
        print(f"  predictor.predict(open('<image>', 'rb').read(), "
              f"initial_args={{'ContentType': 'image/jpeg'}})")


if __name__ == "__main__":
    main()
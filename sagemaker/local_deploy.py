"""SageMaker Local Mode deployment and test script.

Run with: python -m sagemaker.local_deploy [--sample <path>]

Prerequisites:
  - local_train.py has been run (model artifacts exist)
  - Docker installed and Docker socket accessible
"""

import base64
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

    # SageMaker Local Mode は model_data として tar.gz を期待する
    # local_train.py の実行結果 (S3 local file path) を使用
    # 手動デプロイの場合は model.tar.gz を指定
    model_tar = project_root / "model.tar.gz"
    if model_tar.exists():
        model_data = f"file://{model_tar}"
    else:
        # local_train の結果を利用
        model_data = str(project_root / "data" / "model.tar.gz")
        if not Path(model_data.replace("file://", "")).exists():
            logger.error(
                "Model artifact not found. Run local_train first, "
                "or create model.tar.gz from your trained model."
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
        with sample_path.open("rb") as f:
            payload = json.dumps({"image": base64.b64encode(f.read()).decode()})
        result = predictor.predict(payload, initial_args={"ContentType": "application/json"})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.warning("Sample image not found: %s", sample_path)
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        print("Provide --sample <path> to test.")
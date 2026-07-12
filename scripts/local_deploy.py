"""Local deployment and test script.

Run with (ホスト側から実行）:
  python3 -m scripts.local_deploy --sample data/samples/sample_license.jpg

または Docker内から:
  docker compose run --rm -v /var/run/docker.sock:/var/run/docker.sock dev \
      python -m scripts.local_deploy --sample data/samples/sample_license.jpg

デフォルトは `--method serve`（docker compose up serve + HTTP request）。
SageMaker Local Mode を使う場合は `--method sagemaker` を指定。

Prerequisites:
  - モデルが /opt/ml/model/japanese に保存済み
  - Docker installed and Docker socket accessible
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_inside_container() -> bool:
    """コンテナ内で実行されているか判定する。"""
    return Path("/.dockerenv").exists() or os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def _docker_compose(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """docker compose または docker-compose を実行する。"""
    errors = []
    for cmd in [["docker", "compose"] + args, ["docker-compose"] + args]:
        try:
            return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errors.append(str(e))
            continue
    raise RuntimeError(f"Failed to run docker compose {' '.join(args)}: {errors}")


def main() -> None:
    """Deploy and test inference locally."""
    import argparse

    parser = argparse.ArgumentParser(description="Local deploy & test")
    parser.add_argument(
        "--sample",
        default="data/samples/sample_license.jpg",
        help="Sample image path",
    )
    parser.add_argument(
        "--method",
        default="serve",
        choices=["serve", "sagemaker"],
        help="Deployment method: 'serve' (docker compose, default) or 'sagemaker' (Local Mode)",
    )
    args = parser.parse_args()

    if args.method == "sagemaker":
        _deploy_sagemaker_local(args.sample)
    else:
        _deploy_serve(args.sample)


def _deploy_serve(sample_path: str) -> None:
    """docker compose up serve + HTTP request でテストする（推奨方式）。

    ホスト側から実行することを想定。コンテナ内から実行する場合は
    Dockerソケットのマウントとdockerバイナリが必要。
    """
    project_root = Path(__file__).resolve().parent.parent

    # コンテナ内から実行している場合は警告
    if _is_inside_container():
        logger.warning(
            "Running inside a container. docker compose may not be available. "
            "Consider running from host: python3 -m scripts.local_deploy --sample ..."
        )

    # サーバー起動
    logger.info("Starting serve container...")
    _docker_compose(["up", "-d", "serve"], str(project_root))

    # 起動待ち（最大120秒）
    logger.info("Waiting for server to start (max 120s)...")
    for i in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://localhost:8080/ping", timeout=2)
            logger.info("Server is ready!")
            break
        except Exception:
            if i == 59:
                logger.error("Server failed to start within 120 seconds")
                try:
                    _docker_compose(["logs", "--tail=30", "serve"], str(project_root))
                except Exception:
                    pass
                try:
                    _docker_compose(["down"], str(project_root))
                except Exception:
                    pass
                raise RuntimeError("Server startup timeout")
            if i % 5 == 0:
                logger.info("  Waiting... (%d/60)", i + 1)

    # 推論リクエスト送信
    sp = Path(sample_path)
    if not sp.exists():
        logger.warning("Sample image not found: %s", sp)
        print("Server is running at http://localhost:8080")
        print(f"Test with: curl -X POST http://localhost:8080/invocations -H 'Content-Type: image/jpeg' --data-binary @{sp}")
        return

    logger.info("Sending inference request with %s", sp)
    with sp.open("rb") as f:
        raw = f.read()

    req = urllib.request.Request(
        "http://localhost:8080/invocations",
        data=raw,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode("utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error("Inference failed: %s", e)
        try:
            _docker_compose(["logs", "--tail=30", "serve"], str(project_root))
        except Exception:
            pass
        raise
    finally:
        # サーバー停止
        logger.info("Stopping serve container...")
        try:
            _docker_compose(["down"], str(project_root))
        except Exception:
            pass


def _deploy_sagemaker_local(sample_path: str) -> None:
    """SageMaker Local Mode でデプロイしてテストする（AWS認証が必要な場合あり）。"""
    import boto3
    from sagemaker.local import LocalSession
    from sagemaker.pytorch import PyTorchModel

    # AWS認証をダミーで設定
    boto_session = boto3.Session(
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
        region_name="us-east-1",
    )

    # botocoreのクライアントをパッチしてAWS API呼び出しをモック
    import botocore.client
    _orig_client = botocore.client.ClientCreator.create_client
    def _patched_create_client(self, service_name, *args, **kwargs):
        client = _orig_client(self, service_name, *args, **kwargs)
        if service_name == "sts":
            client.get_caller_identity = lambda: {"Account": "000000000000"}
        if service_name == "s3":
            client.list_buckets = lambda: {"Buckets": []}
            client.head_bucket = lambda Bucket=None, **kw: {}
        return client
    botocore.client.ClientCreator.create_client = _patched_create_client

    sagemaker_session = LocalSession(boto_session=boto_session)
    sagemaker_session.config = {"local": {"local_code": True, "region_name": "us-east-1"}}

    project_root = Path(__file__).resolve().parent.parent

    # model.tar.gz を探す
    model_tar = project_root / "model.tar.gz"
    if not model_tar.exists():
        logger.error(
            "model.tar.gz not found. Create it first:\n"
            "  docker compose run --rm dev bash -c "
            "\"cd /opt/ml/model/japanese && tar czf /opt/ml/code/model.tar.gz --exclude='checkpoint-*' .\""
        )
        raise FileNotFoundError("model.tar.gz not found")

    model = PyTorchModel(
        model_data=f"file://{model_tar}",
        role="arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-Dummy",
        entry_point="sagemaker_serve/inference_entry_point.py",
        source_dir=str(project_root),
        image_uri="yomitori:infer",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
    )

    sp = Path(sample_path)
    if sp.exists():
        with sp.open("rb") as f:
            raw = f.read()
        result = predictor.predict(
            raw,
            initial_args={"ContentType": "image/jpeg"},
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.warning("Sample image not found: %s", sp)
        print(f"Endpoint deployed: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
"""Local deployment and test script.

Run with:
  docker compose run --rm dev python -m scripts.local_deploy --sample data/samples/sample_license.jpg

または ホスト側から:
  python3 -m scripts.local_deploy --sample data/samples/sample_license.jpg

2つの方式をサポート:
  --method sagemaker (デフォルト): SageMaker Local Mode でエンドポイント構築
  --method serve: docker compose up serve + HTTP request

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
    return Path("/.dockerenv").exists()


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
        default="sagemaker",
        choices=["sagemaker", "serve"],
        help="Deployment method: 'sagemaker' (Local Mode, default) or 'serve' (docker compose)",
    )
    args = parser.parse_args()

    if args.method == "sagemaker":
        _deploy_sagemaker_local(args.sample)
    else:
        _deploy_serve(args.sample)


def _deploy_sagemaker_local(sample_path: str) -> None:
    """SageMaker Local Mode でデプロイしてテストする。"""
    import boto3
    from sagemaker.local import LocalSession

    boto_session = boto3.Session(
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
        region_name="us-east-1",
    )

    # botocoreのクライアントをパッチしてAWS API呼び出しをモック
    import botocore.client
    _orig_create_client = botocore.client.ClientCreator.create_client

    def _patched_create_client(self, service_name, *args, **kwargs):
        client = _orig_create_client(self, service_name, *args, **kwargs)
        if service_name == "sts":
            client.get_caller_identity = lambda: {"Account": "000000000000"}
        if service_name == "s3":
            client.list_buckets = lambda: {"Buckets": []}
            client.head_bucket = lambda Bucket=None, **kw: {}
            client.head_object = lambda Bucket=None, Key=None, **kw: {}
            # S3アップロードをローカルファイルコピーに置換
            def _mock_upload(Filename, Bucket, Key, **kw):
                local_s3 = Path(f"/tmp/s3_mock/{Bucket}/{Key}")
                local_s3.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(Filename, str(local_s3))
                logger.info("Mock S3 upload: %s → s3://%s/%s", Filename, Bucket, Key)
            client.upload_file = _mock_upload
            # S3ダウンロードをローカルファイルコピーに置換
            def _mock_download(Bucket, Key, Filename, **kw):
                local_s3 = Path(f"/tmp/s3_mock/{Bucket}/{Key}")
                if local_s3.exists():
                    import shutil
                    shutil.copy2(str(local_s3), Filename)
                    logger.info("Mock S3 download: s3://%s/%s → %s", Bucket, Key, Filename)
                else:
                    logger.warning("Mock S3 file not found: s3://%s/%s", Bucket, Key)
            client.download_file = _mock_download
        return client

    botocore.client.ClientCreator.create_client = _patched_create_client

    sagemaker_session = LocalSession(boto_session=boto_session)
    sagemaker_session.config = {"local": {"local_code": True, "region_name": "us-east-1"}}

    project_root = Path(__file__).resolve().parent.parent

    # model.tar.gz を探すまたは作成
    model_tar = project_root / "model.tar.gz"
    if not model_tar.exists():
        # /opt/ml/model/japanese から model.tar.gz を作成
        model_dir = Path("/opt/ml/model/japanese")
        if not model_dir.exists():
            # フォールバック: /opt/ml/model 直下
            model_dir = Path("/opt/ml/model")
        if model_dir.exists():
            logger.info("Creating model.tar.gz from %s ...", model_dir)
            subprocess.run(
                ["tar", "czf", str(model_tar), "--exclude=checkpoint-*", "."],
                cwd=str(model_dir),
                check=True,
            )
        else:
            logger.error(
                "model.tar.gz not found and no model directory exists.\n"
                "Run fine-tuning first:\n"
                "  docker compose run --rm train python -m training.train_trocr ..."
            )
            raise FileNotFoundError("model.tar.gz not found")

    logger.info("Deploying SageMaker Local Mode endpoint...")
    # カスタムイメージのCMDをそのまま使うため Model クラスを使用
    from sagemaker.model import Model

    model = Model(
        image_uri="yomitori:infer",
        model_data=f"file://{model_tar}",
        role="arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-Dummy",
        env={
            "SAGEMAKER_MODEL_DIR": "/opt/ml/model",
            "YOMITORI_MODEL_DIR": "/opt/ml/model",
        },
        sagemaker_session=sagemaker_session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
    )

    sp = Path(sample_path)
    if sp.exists():
        with sp.open("rb") as f:
            raw = f.read()
        logger.info("Sending inference request with %s ...", sp)
        result = predictor.predict(
            raw,
            initial_args={"ContentType": "image/jpeg"},
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.warning("Sample image not found: %s", sp)
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        print("Test with:")
        print(f"  predictor.predict(open('<image>', 'rb').read(), "
              f"initial_args={{'ContentType': 'image/jpeg'}})")


def _deploy_serve(sample_path: str) -> None:
    """docker compose up serve + HTTP request でテストする。"""
    project_root = Path(__file__).resolve().parent.parent

    if _is_inside_container():
        logger.warning(
            "Running inside a container. docker compose requires Docker socket mount."
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
        logger.info("Stopping serve container...")
        try:
            _docker_compose(["down"], str(project_root))
        except Exception:
            pass


if __name__ == "__main__":
    main()
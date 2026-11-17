import hashlib
import shutil
from pathlib import Path

import kagglehub

from credit_scoring.config import KAGGLE_DATASET, RAW_FILE_NAME, RAW_FILE_SHA256, RAW_PATH
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_cached() -> bool:
    return RAW_PATH.exists() and sha256_of(RAW_PATH) == RAW_FILE_SHA256


@tracked_step("download")
def download_raw_dataset() -> None:
    log = get_pipeline_logger("download")
    if is_cached():
        log.info("raw_dataset_cached", extra={"path": str(RAW_PATH)})
        return
    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / RAW_FILE_NAME
    checksum = sha256_of(downloaded)
    if checksum != RAW_FILE_SHA256:
        log.error("raw_dataset_checksum_mismatch", extra={"sha256": checksum})
        raise SystemExit(1)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, RAW_PATH)
    log.info("raw_dataset_downloaded", extra={"source": KAGGLE_DATASET, "path": str(RAW_PATH)})


if __name__ == "__main__":
    download_raw_dataset()

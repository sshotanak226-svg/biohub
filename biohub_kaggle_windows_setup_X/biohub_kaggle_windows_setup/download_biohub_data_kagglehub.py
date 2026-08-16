"""Download and validate the Biohub competition with the official kagglehub API."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

# Set the documented kagglehub cache override before importing kagglehub.
# Keeping it inside the repository makes interrupted downloads resumable and
# keeps the final train/test data at a predictable location.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE_ROOT / "biohub_top10_download" / "data" / "kagglehub_cache"
os.environ.setdefault("KAGGLEHUB_CACHE", str(CACHE_ROOT))

import kagglehub  # noqa: E402
import requests  # noqa: E402
from kagglehub import clients as kagglehub_clients  # noqa: E402

COMPETITION = "biohub-cell-tracking-during-development"
EXPECTED_COUNTS = {
    "train": 24_477,
    "test": 408,
    "sample_submission.csv": 1,
}
EXPECTED_UNCOMPRESSED_BYTES = 87_609_892_618
READ_TIMEOUT_SECONDS = 300
RETRY_DELAY_SECONDS = 60
LEGACY_DIRECTORIES = (
    "competition",
    "kagglehub_official",
    "kagglehub_probe",
)


def validate_extracted(destination: Path) -> dict[str, object]:
    files = [path for path in destination.rglob("*") if path.is_file()]
    relative = [path.relative_to(destination).as_posix() for path in files]
    counts = Counter(
        name.split("/", 1)[0] if "/" in name else name for name in relative
    )
    total_bytes = sum(path.stat().st_size for path in files)

    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"file-count mismatch: {dict(counts)} != {EXPECTED_COUNTS}"
        )
    if total_bytes != EXPECTED_UNCOMPRESSED_BYTES:
        raise RuntimeError(
            f"extracted-size mismatch: {total_bytes} != {EXPECTED_UNCOMPRESSED_BYTES}"
        )

    return {
        "status": "ok",
        "download_method": "kagglehub.competition_download",
        "kagglehub_version": kagglehub.__version__,
        "competition": COMPETITION,
        "destination": str(destination),
        "file_count": len(files),
        "counts": dict(counts),
        "uncompressed_bytes": total_bytes,
        "evaluation_data_note": (
            "Kaggle distributes test inputs and sample_submission.csv; "
            "test ground-truth labels are not distributed."
        ),
    }


def remove_legacy_downloads(destination: Path) -> list[str]:
    data_root = (WORKSPACE_ROOT / "biohub_top10_download" / "data").resolve()
    official_destination = destination.resolve()
    removed: list[str] = []
    for name in LEGACY_DIRECTORIES:
        target = (data_root / name).resolve()
        if target.parent != data_root or target == official_destination:
            raise RuntimeError(f"unsafe legacy cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed


def main() -> None:
    # kagglehub 1.0.2 defaults to a 15-second read timeout and lets a transient
    # requests ConnectionError terminate a large competition download. Extend
    # the timeout and call the public API again; kagglehub then resumes the same
    # archive using an HTTP Range request.
    kagglehub_clients.DEFAULT_READ_TIMEOUT = READ_TIMEOUT_SECONDS
    destination: Path | None = None
    attempt = 0
    while destination is None:
        attempt += 1
        try:
            print(f"KAGGLEHUB_ATTEMPT={attempt}", flush=True)
            destination = Path(
                kagglehub.competition_download(COMPETITION)
            ).resolve()
        except requests.RequestException as exc:
            print(
                f"KAGGLEHUB_RETRY={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    report = validate_extracted(destination)
    report["removed_legacy_directories"] = remove_legacy_downloads(destination)
    report_path = (
        WORKSPACE_ROOT
        / "biohub_top10_download"
        / "data"
        / "kagglehub_validation.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"KAGGLEHUB_COMPLETE={destination}", flush=True)
    print(f"VALIDATION_REPORT={report_path}", flush=True)


if __name__ == "__main__":
    main()

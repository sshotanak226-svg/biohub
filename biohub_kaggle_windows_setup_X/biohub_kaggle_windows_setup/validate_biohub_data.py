"""Validate the official Biohub competition archive and its extraction."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED_ARCHIVE_SIZE = 87_393_127_165
EXPECTED_FILE_COUNTS = {
    "train": 24_477,
    "test": 408,
    "sample_submission.csv": 1,
}


def archive_inventory(archive: Path, check_crc: bool) -> tuple[list[zipfile.ZipInfo], dict]:
    if not archive.is_file():
        raise RuntimeError(f"archive not found: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError(
            f"archive size mismatch: {actual_size} != {EXPECTED_ARCHIVE_SIZE}"
        )

    started = time.time()
    with zipfile.ZipFile(archive) as zipped:
        files = [item for item in zipped.infolist() if not item.is_dir()]
        counts = Counter(
            item.filename.split("/", 1)[0]
            if "/" in item.filename
            else item.filename
            for item in files
        )
        if dict(counts) != EXPECTED_FILE_COUNTS:
            raise RuntimeError(
                f"archive inventory mismatch: {dict(counts)} != {EXPECTED_FILE_COUNTS}"
            )
        bad_member = zipped.testzip() if check_crc else None
        if bad_member is not None:
            raise RuntimeError(f"CRC check failed: {bad_member}")

    report = {
        "archive": str(archive.resolve()),
        "archive_size_bytes": actual_size,
        "archive_file_count": len(files),
        "archive_uncompressed_bytes": sum(item.file_size for item in files),
        "archive_counts": dict(counts),
        "crc_checked": check_crc,
        "crc_ok": True if check_crc else None,
        "archive_validation_seconds": round(time.time() - started, 3),
    }
    return files, report


def validate_extraction(files: list[zipfile.ZipInfo], destination: Path) -> dict:
    expected = {item.filename: item.file_size for item in files}
    actual: dict[str, int] = {}
    for root_name in ("train", "test"):
        root = destination / root_name
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    actual[path.relative_to(destination).as_posix()] = path.stat().st_size
    sample = destination / "sample_submission.csv"
    if sample.is_file():
        actual[sample.name] = sample.stat().st_size

    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    wrong_size = sorted(
        name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
    )
    if missing or extra or wrong_size:
        preview = {
            "missing": missing[:10],
            "extra": extra[:10],
            "wrong_size": wrong_size[:10],
        }
        raise RuntimeError(f"extracted data mismatch: {preview}")

    counts = Counter(
        name.split("/", 1)[0] if "/" in name else name for name in actual
    )
    return {
        "destination": str(destination.resolve()),
        "extracted_file_count": len(actual),
        "extracted_bytes": sum(actual.values()),
        "extracted_counts": dict(counts),
        "extracted_sizes_match_archive": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--crc", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        files, report = archive_inventory(args.archive, args.crc)
        if args.destination is not None:
            report.update(validate_extraction(files, args.destination))
        report["status"] = "ok"
        report["evaluation_data_note"] = (
            "The competition distributes test inputs and sample_submission.csv; "
            "test ground-truth labels are not included."
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Biohub data validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

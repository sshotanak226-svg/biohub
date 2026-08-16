"""Read-only L0 integrity check for the locally downloaded competition archive."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def inspect_archive(path: Path) -> dict:
    partials = sorted(str(item) for item in path.parent.glob("*.kaggle-partial"))
    result = {
        "archive": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else 0,
        "partial_markers": partials, "central_directory_valid": False, "entry_count": 0,
        "zarr_entry_count": 0, "geff_entry_count": 0,
    }
    if not path.is_file() or partials:
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            result["entry_count"] = len(infos)
            result["zarr_entry_count"] = sum(".zarr/" in info.filename for info in infos)
            result["geff_entry_count"] = sum(".geff/" in info.filename for info in infos)
            result["central_directory_valid"] = bool(infos)
    except (OSError, zipfile.BadZipFile) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Biohub ZIP central directory without extraction")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    result = inspect_archive(args.archive)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["central_directory_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Resume the official competition ZIP with validated parallel HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

COMPETITION = "biohub-cell-tracking-during-development"
REPORT_LOCK = threading.Lock()


def get_download() -> tuple[requests.Response, str, dict[str, str], int, str]:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = COMPETITION
        response = kaggle.competitions.competition_api_client.download_data_files(request)
    total = int(response.headers["Content-Length"])
    validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
    if response.headers.get("Accept-Ranges") != "bytes" or not validator:
        response.close()
        raise RuntimeError("The Kaggle response is not safely range-resumable")
    return response, response.url, dict(response.request.headers), total, validator


def download_range(
    part_path: Path,
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    retries: int,
) -> int:
    existing = part_path.stat().st_size if part_path.is_file() else 0
    position = start + existing
    if position > end + 1:
        raise RuntimeError(f"oversized range part: {part_path}")
    session = requests.Session()
    attempt = 0
    while position <= end:
        attempt += 1
        request_headers = dict(headers)
        request_headers["Range"] = f"bytes={position}-{end}"
        try:
            with session.get(url, headers=request_headers, stream=True, timeout=(30, 300)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range {position}-{end}: HTTP {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {position}-"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                with part_path.open("ab", buffering=0) as output:
                    checkpoint = position + 64 * 1024 * 1024
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        position += len(chunk)
                        if position >= checkpoint:
                            output.flush()
                            checkpoint = position + 64 * 1024 * 1024
                attempt = 0
        except (OSError, requests.RequestException, RuntimeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"range {start}-{end} failed at {position}: {exc}") from exc
            with REPORT_LOCK:
                print(f"Range {start}-{end} retry {attempt}/{retries} at {position}: {exc}", flush=True)
            time.sleep(min(30, 2**attempt))
    return position


def copy_exact(source, destination, length: int) -> None:
    remaining = length
    while remaining:
        block = source.read(min(16 * 1024 * 1024, remaining))
        if not block:
            raise RuntimeError("unexpected end of file while assembling ZIP")
        destination.write(block)
        remaining -= len(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()
    archive = args.archive.resolve()
    marker_path = Path(str(archive) + ".kaggle-partial")
    state_dir = Path(str(archive) + ".parallel-state")
    state_dir.mkdir(exist_ok=True)

    response, url, headers, total, validator = get_download()
    response.close()
    if not marker_path.is_file():
        raise RuntimeError(f"resume marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if int(marker.get("size", -1)) != total or marker.get("validator") != validator:
        raise RuntimeError("local partial marker does not match the current Kaggle object")
    if total != 87_393_127_165:
        raise RuntimeError(f"unexpected competition ZIP size: {total}")

    state_path = state_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prefix = int(state["prefix"])
        ranges = [tuple(map(int, item)) for item in state["ranges"]]
        if int(state["total"]) != total or state["validator"] != validator:
            raise RuntimeError("parallel state does not match the current Kaggle object")
    else:
        prefix = archive.stat().st_size
        if not 0 < prefix < total:
            raise RuntimeError(f"invalid partial size: {prefix}")
        remaining = total - prefix
        ranges = []
        cursor = prefix
        for index in range(args.workers):
            length = remaining // args.workers + (1 if index < remaining % args.workers else 0)
            ranges.append((cursor, cursor + length - 1))
            cursor += length
        state_path.write_text(json.dumps({
            "total": total, "validator": validator, "prefix": prefix, "ranges": ranges
        }, indent=2), encoding="utf-8")
    part_paths = [state_dir / f"range_{index:02d}.part" for index in range(len(ranges))]
    active_workers = min(args.workers, len(ranges))
    print(
        f"Official ZIP: {total} bytes; verified prefix: {prefix} bytes; "
        f"ranges: {len(ranges)}; active workers: {active_workers}",
        flush=True,
    )
    stop_reporter = threading.Event()

    def report() -> None:
        while not stop_reporter.wait(30):
            completed = prefix + sum(path.stat().st_size if path.is_file() else 0 for path in part_paths)
            print(f"Progress: {completed / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB ({100*completed/total:.2f}%)", flush=True)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=active_workers) as executor:
            futures = [
                executor.submit(download_range, path, url, headers, start, end, args.retries)
                for (start, end), path in zip(ranges, part_paths)
            ]
            for future, (_, end) in zip(futures, ranges):
                if future.result() != end + 1:
                    raise RuntimeError("a range ended at an unexpected position")
    finally:
        stop_reporter.set()
        reporter.join()

    for path, (start, end) in zip(part_paths, ranges):
        expected = end - start + 1
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"range part has the wrong size: {path}")

    assembled = Path(str(archive) + ".parallel-complete")
    with assembled.open("wb", buffering=16 * 1024 * 1024) as output:
        with archive.open("rb", buffering=16 * 1024 * 1024) as source:
            copy_exact(source, output, prefix)
        for path in part_paths:
            with path.open("rb", buffering=16 * 1024 * 1024) as source:
                copy_exact(source, output, path.stat().st_size)
        output.flush()
        os.fsync(output.fileno())
    if assembled.stat().st_size != total:
        raise RuntimeError("assembled ZIP has the wrong final size")
    os.replace(assembled, archive)
    marker_path.unlink()
    for path in part_paths:
        path.unlink(missing_ok=True)
    state_path.unlink()
    state_dir.rmdir()
    print(f"Parallel download complete: {archive} ({total} bytes)", flush=True)


if __name__ == "__main__":
    main()

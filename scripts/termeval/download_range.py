"""Resumable HTTP range download for the open term-evaluation corpora."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def _size(url: str) -> int:
    response = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    response.raise_for_status()
    if response.status_code != 206 or "Content-Range" not in response.headers:
        raise RuntimeError("source does not support byte ranges")
    return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def _part(url: str, path: Path, start: int, end: int) -> None:
    wanted = end - start + 1
    for attempt in range(8):
        have = path.stat().st_size if path.exists() else 0
        if have == wanted:
            return
        if have > wanted:
            raise RuntimeError(f"oversized part: {path}")
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start + have}-{end}"},
                stream=True,
                timeout=(20, 90),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected 206, got {response.status_code}")
                expected = f"bytes {start + have}-{end}/"
                if not response.headers.get("Content-Range", "").startswith(expected):
                    raise RuntimeError("server returned a different byte range")
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size == wanted:
                return
        except (OSError, requests.RequestException) as exc:
            if attempt == 7:
                raise RuntimeError(f"part {path.name} failed") from exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"incomplete part: {path}")


def download(url: str, output: Path, parts: int, md5: str | None) -> None:
    size = _size(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == size:
        print(f"Already downloaded: {output} ({size} bytes)")
    else:
        part_dir = output.with_name(output.name + ".parts")
        part_dir.mkdir(exist_ok=True)
        chunk_size = math.ceil(size / parts)
        first_part = part_dir / "000.part"
        if output.exists():
            if output.stat().st_size <= chunk_size and not first_part.exists():
                output.rename(first_part)
            else:
                raise RuntimeError("existing partial download cannot be resumed safely")
        jobs = []
        with ThreadPoolExecutor(max_workers=parts) as pool:
            for index in range(parts):
                start = index * chunk_size
                end = min(size - 1, start + chunk_size - 1)
                if start <= end:
                    jobs.append(pool.submit(_part, url, part_dir / f"{index:03d}.part", start, end))
            for future in as_completed(jobs):
                future.result()
        assembled = output.with_name(output.name + ".assembled")
        with assembled.open("wb") as target:
            for index in range(parts):
                source = part_dir / f"{index:03d}.part"
                with source.open("rb") as handle:
                    while block := handle.read(1 << 20):
                        target.write(block)
        if assembled.stat().st_size != size:
            raise RuntimeError("assembled download size mismatch")
        os.replace(assembled, output)
        for source in part_dir.glob("*.part"):
            source.unlink()
        part_dir.rmdir()
        print(f"Downloaded: {output} ({size} bytes)")
    if md5:
        digest = hashlib.md5()
        with output.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        if digest.hexdigest() != md5:
            raise RuntimeError(f"MD5 mismatch: {output}")
        print(f"MD5 verified: {digest.hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--md5")
    args = parser.parse_args()
    download(args.url, args.output, args.parts, args.md5)

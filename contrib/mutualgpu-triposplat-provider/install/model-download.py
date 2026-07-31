#!/usr/bin/env python3
"""Download and verify the exact non-secret TripoSplat model manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repository, revision, files = manifest["repository"], manifest["revision"], manifest["files"]
    for item in files:
        relative = item["path"]
        destination = args.model_dir / relative
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{relative}?download=true"
        if args.dry_run:
            print(f"would fetch {relative} from pinned revision {revision}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if valid(destination, item):
            print(f"verified existing {relative}")
            continue
        with tempfile.NamedTemporaryFile(prefix=".download-", dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                request = Request(url, headers={"User-Agent": "mutualgpu-triposplat-installer/0.1"})
                with urlopen(request, timeout=90) as response:
                    digest = hashlib.sha256()
                    length = 0
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)
                        digest.update(chunk)
                        length += len(chunk)
                if length != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"download verification failed for {relative}")
                os.chmod(temporary_path, 0o600)
                temporary_path.replace(destination)
                print(f"downloaded and verified {relative}")
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise


def valid(path: Path, item: dict[str, object]) -> bool:
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


if __name__ == "__main__":
    main()


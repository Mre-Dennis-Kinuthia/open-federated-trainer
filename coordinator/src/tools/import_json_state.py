#!/usr/bin/env python3
"""Import existing JSON/model files into the artifact manifest index.

Does not delete source files. Idempotent by content hash.

Usage from coordinator/src:

  python -m tools.import_json_state
  python -m tools.import_json_state --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

COORD_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import JSON state into artifact index")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=COORD_ROOT / "models",
        help="Directory of model_v*.json files",
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=COORD_ROOT.parent / "adapters",
        help="Directory of adapter model JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing",
    )
    args = parser.parse_args(argv)

    from artifacts import get_artifact_store
    from persistence import ArtifactRecord
    from persistence.json_repos import get_artifact_repository, sha256_file

    store = get_artifact_store()
    repo = get_artifact_repository()
    imported = 0

    for directory, artifact_type in (
        (args.models_dir, "global_model"),
        (args.adapters_dir, "lora_adapter"),
    ):
        if not directory.exists():
            print(f"skip missing dir: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            content_hash = sha256_file(path)
            artifact_id = f"{artifact_type}:{path.stem}:{content_hash[:12]}"
            existing = repo.get_by_hash(content_hash)
            if existing:
                print(f"exists {artifact_id}")
                continue
            if args.dry_run:
                print(f"would import {path} -> {artifact_id}")
                imported += 1
                continue
            stored_hash = store.put_file(path)
            assert stored_hash == content_hash
            repo.put_manifest(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    content_hash=content_hash,
                    byte_size=path.stat().st_size,
                    storage_uri=store.uri_for(content_hash),
                    media_type="application/json",
                    created_at=time.time(),
                    metadata={"source_path": str(path)},
                )
            )
            print(f"imported {artifact_id}")
            imported += 1

    data_dir = COORD_ROOT / "data"
    for name in ("state.json", "jobs.json", "lora_rounds.json", "geo_presence.json"):
        path = data_dir / name
        status = "present" if path.exists() else "missing"
        print(f"control {name}: {status}")

    print(json.dumps({"imported": imported, "dry_run": args.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

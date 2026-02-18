#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_dentistry.config import load_protocol
from ai_dentistry.pubmed import fetch_query_snapshot


def load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch PubMed snapshots for all configured study queries."
    )
    parser.add_argument(
        "--protocol",
        default="config/protocol.yaml",
        help="Path to protocol YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where raw query snapshots are written.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing pubmed_records_*.jsonl files in output-dir before fetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env_file(PROJECT_ROOT / ".env")
    protocol = load_protocol(args.protocol)
    entrez_cfg = protocol.get("entrez", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for file in output_dir.glob("pubmed_records_*.jsonl"):
            file.unlink()

    for query in protocol["queries"]:
        query_id = query["id"]
        query_text = query["query"]
        out_path = fetch_query_snapshot(
            query_id=query_id,
            query_text=query_text,
            entrez_cfg=entrez_cfg,
            output_dir=output_dir,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

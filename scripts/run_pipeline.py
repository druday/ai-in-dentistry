#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_dentistry.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full AI-in-Dentistry temporal network analysis pipeline."
    )
    parser.add_argument(
        "--protocol",
        default="config/protocol.yaml",
        help="Path to protocol YAML file.",
    )
    parser.add_argument(
        "--raw-glob",
        default="data/raw/pubmed_records_*.jsonl",
        help="Glob pattern for raw PubMed snapshots.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--period-mode",
        default=None,
        choices=["fixed", "balanced", "dynamic"],
        help=(
            "Override temporal segmentation mode from protocol. "
            "Use 'fixed' for hardcoded bins or 'balanced'/'dynamic' for auto-balanced bins."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing generated graph/html/centrality files before writing new outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        protocol_path=args.protocol,
        raw_glob=args.raw_glob,
        project_root=args.project_root,
        period_mode_override=args.period_mode,
        clean_output_dirs=args.clean_output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

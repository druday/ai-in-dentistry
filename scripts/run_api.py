#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastAPI backend for the reproducibility UI.")
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Optional project root used by the API for protocol/data/outputs. "
            "Defaults to this repository root."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", default=8000, type=int, help="Port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.project_root:
        os.environ["AI_DENTISTRY_PROJECT_ROOT"] = str(Path(args.project_root).resolve())
    uvicorn.run(
        "ai_dentistry.api.app:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
    )


if __name__ == "__main__":
    main()

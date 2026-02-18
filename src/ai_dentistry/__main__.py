from __future__ import annotations

import json

from ai_dentistry.pipeline import run_pipeline


def main() -> None:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

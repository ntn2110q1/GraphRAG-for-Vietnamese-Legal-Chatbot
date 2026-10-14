"""Evaluate saved trainer loss metrics; this does not claim answer correctness."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="training_metrics.json produced by train_qlora.py")
    args = parser.parse_args()
    value = json.loads(args.metrics.read_text(encoding="utf-8"))
    print(json.dumps(value, ensure_ascii=False, indent=2))
    print("These are model loss metrics only. Evaluate legal answers and citations against an independently labeled set.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run end-to-end AI supply chain decision support pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supply_chain_ai.pipeline import SupplyChainDecisionSupportSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supply chain AI decision support pipeline.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional path to historical supply chain CSV. If omitted, synthetic data is generated.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for artifacts.")
    parser.add_argument(
        "--synthetic-periods",
        type=int,
        default=730,
        help="Number of daily periods to generate when input CSV is not provided.",
    )
    parser.add_argument(
        "--no-copilot",
        action="store_true",
        help="Disable RAG copilot indexing and sample response generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = SupplyChainDecisionSupportSystem(random_state=42)
    results = system.run(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        synthetic_periods=args.synthetic_periods,
        build_copilot=not args.no_copilot,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

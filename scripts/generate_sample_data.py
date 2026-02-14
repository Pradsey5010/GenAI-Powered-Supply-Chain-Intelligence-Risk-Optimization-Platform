#!/usr/bin/env python3
"""Generate synthetic historical supply chain dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supply_chain_ai.data import SyntheticDataConfig, generate_synthetic_supply_chain_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic supply chain data.")
    parser.add_argument("--output-path", type=str, default="data/historical_sales.csv")
    parser.add_argument("--periods", type=int, default=730)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticDataConfig(periods=args.periods)
    frame = generate_synthetic_supply_chain_data(config=config)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} records to {output}")


if __name__ == "__main__":
    main()

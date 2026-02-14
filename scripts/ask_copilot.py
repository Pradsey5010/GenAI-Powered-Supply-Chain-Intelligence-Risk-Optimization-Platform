#!/usr/bin/env python3
"""Ask a question to the supply chain RAG copilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supply_chain_ai.rag import EmbeddingIndex, SupplyChainCopilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the supply chain copilot.")
    parser.add_argument("--index-path", type=str, default="outputs/copilot/embedding_index.pkl")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = EmbeddingIndex.load(args.index_path)
    copilot = SupplyChainCopilot(index=index)
    response = copilot.ask(question=args.question, top_k=args.top_k)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

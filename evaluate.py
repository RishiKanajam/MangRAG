"""
Evaluation CLI for the RAG pipeline.

Usage:
    python evaluate.py eval_set.jsonl
    python evaluate.py eval_set.jsonl --k 3

eval_set.jsonl — one JSON object per line:
    {"query": "What is mangrove zonation?", "relevant_texts": ["Mangrove zonation refers to..."]}

relevant_texts: short reference passages or key phrases that a correct answer should cover.
Precision@k counts how many retrieved chunks contain at least one reference passage.
Faithfulness is an LLM-as-judge score (0–1) of whether the generated answer is grounded in context.
"""
import sys
import json
import argparse
import dotenv
dotenv.load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from mangrag.eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with precision@k and faithfulness"
    )
    parser.add_argument("eval_set", help="JSONL file — one {query, relevant_texts} per line")
    parser.add_argument("--k", type=int, default=5, help="Retrieval depth (default: 5)")
    args = parser.parse_args()

    try:
        with open(args.eval_set) as f:
            queries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: file not found — {args.eval_set}")
        sys.exit(1)

    if not queries:
        print("No queries found in eval set.")
        sys.exit(0)

    results = []
    for i, item in enumerate(queries, 1):
        q = item.get('query', '')
        refs = item.get('relevant_texts', [])
        print(f"[{i}/{len(queries)}] {q[:70]}...")
        result = evaluate(q, refs, k=args.k)
        results.append(result)
        print(
            f"  precision@{args.k}={result[f'precision@{args.k}']:.3f}  "
            f"faithfulness={result['faithfulness']:.3f}"
        )

    k = args.k
    avg_p = sum(r[f'precision@{k}'] for r in results) / len(results)
    avg_f = sum(r['faithfulness'] for r in results) / len(results)
    print(f"\n{'─' * 50}")
    print(f"Queries evaluated : {len(results)}")
    print(f"Mean precision@{k} : {avg_p:.3f}")
    print(f"Mean faithfulness : {avg_f:.3f}")

    out_path = args.eval_set.replace('.jsonl', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()

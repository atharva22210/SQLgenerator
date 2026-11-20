"""Build/search/evaluate V1: python -m rag.cli --help."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from rag.bm25 import BM25Index, read_catalog, sha256


def evaluate(index: BM25Index, labels: Path, *, k: int, scoped: bool) -> dict:
    """Evaluate fixed questions without sending gold table names to search.

    scoped=True supplies the application-selected database (an easier task).
    scoped=False searches all databases, receiving ONLY the English question.
    Gold IDs are read only AFTER retrieval for scoring. This function does not
    measure SQL execution accuracy because it does not call the SQL generator.
    """
    tasks = [json.loads(line) for line in labels.read_text().splitlines()]
    if not tasks:
        raise ValueError("Evaluation set is empty")
    records = []
    for task in tasks:
        response = index.search(task["question"], k=k, db_id=task["db_id"] if scoped else None)
        predicted = [r["document_id"] for r in response["results"]]
        gold = set(task["gold_document_ids"])
        if not gold:
            raise ValueError("Expected at least one gold table per task")
        found = gold & set(predicted)
        first_rank = next((i for i, doc in enumerate(predicted, 1) if doc in gold), None)
        records.append({"id": task["id"], "question": task["question"], "gold": sorted(gold),
                        "predicted": predicted, "recall": len(found) / len(gold),
                        # Precision@k uses requested k: a short result list
                        # does not receive artificially perfect precision.
                        "precision_at_k": len(found) / k, "complete": int(found == gold),
                        "reciprocal_rank": 1 / first_rank if first_rank else 0,
                        "multi_table": len(gold) > 1, "elapsed_ms": response["elapsed_ms"]})
    def summarize(rows):
        if not rows:
            return {"questions": 0}
        timings = sorted(row["elapsed_ms"] for row in rows)
        return {"questions": len(rows), **{metric: statistics.mean(row[metric] for row in rows)
                for metric in ("recall", "precision_at_k", "complete", "reciprocal_rank")},
                "p50_ms": statistics.median(timings), "p95_ms": timings[math.ceil(.95 * len(timings)) - 1]}
    return {"method": "bm25", "k": k, "scope": "database" if scoped else "global",
            "source_sha256": index.source_sha256, "k1": index.k1, "b": index.b,
            "labels_sha256": sha256(labels),
            "all": summarize(records), "multi_table": summarize([r for r in records if r["multi_table"]]),
            "latency_scope": "in-process retrieval only; excludes startup/network/generation",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Completed RAG bundle directory")
    parser.add_argument("--index", type=Path, required=True, help="Versioned JSON schema index")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build schema-only index; no model training")
    build.add_argument("--k1", type=float, default=1.2)
    build.add_argument("--b", type=float, default=.75)
    search = commands.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--db-id", help="Optional application-selected database")
    search.add_argument("--explain", action="store_true")
    search.add_argument("--k", type=int, default=5)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--scoped", action="store_true", help="Supply known database; easier than routing")
    evaluation.add_argument("--k", type=int, default=5)
    evaluation.add_argument("--output", type=Path, required=True, help="New report JSON file")
    args = parser.parse_args()
    try:
        if args.command == "build":
            started = time.perf_counter()
            docs, checksum = read_catalog(args.bundle)
            index = BM25Index(docs, checksum, k1=args.k1, b=args.b)
            index.save(args.index)
            result = {"index": str(args.index), "tables": len(docs), "source_sha256": checksum,
                      "build_ms": (time.perf_counter() - started) * 1000}
        else:
            index = BM25Index.load(args.index, args.bundle)
            if args.command == "search":
                result = index.search(args.query, k=args.k, db_id=args.db_id, explain=args.explain)
            else:
                result = evaluate(index, args.bundle / f"labels/{args.split}.jsonl", k=args.k, scoped=args.scoped)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as file:
                    json.dump(result, file, indent=2)
                result = {key: value for key, value in result.items() if key != "records"}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"RAG error: {error}\n")


if __name__ == "__main__":
    main()

"""Shared seed retrieval metrics and final-context coverage for all methods."""
import json
import math
from pathlib import Path
import statistics
from rag.common.catalog import sha256
from rag.common.search import Retriever


def evaluate(index: Retriever, labels: Path, *, k: int, scoped: bool) -> dict:
    """Evaluate fixed questions without sending gold table names to search.

    scoped=True supplies the application-selected database (an easier task).
    scoped=False searches all databases, receiving ONLY the English question.
    Gold IDs are read only AFTER retrieval for scoring. This function does not
    measure SQL execution accuracy because it does not call the SQL generator.
    For graph retrieval k counts seeds: preserve top-k metrics on seed_results,
    and measure expanded schemas separately using context_* metrics. Precision
    of an expanded context divides by its actual size, so it cannot exceed one.
    """
    tasks = [json.loads(line) for line in labels.read_text().splitlines()]
    if not tasks:
        raise ValueError("Evaluation set is empty")
    records = []
    for task in tasks:
        response = index.search(task["question"], k=k, db_id=task["db_id"] if scoped else None)
        predicted = [r["document_id"] for r in response.get("seed_results", response["results"])]
        context_ids = [r["document_id"] for r in response["results"]]
        gold = set(task["gold_document_ids"])
        if not gold:
            raise ValueError("Expected at least one gold table per task")
        found = gold & set(predicted)
        context_found = gold & set(context_ids)
        first_rank = next((i for i, doc in enumerate(predicted, 1) if doc in gold), None)
        records.append({"id": task["id"], "question": task["question"], "gold": sorted(gold),
                        "predicted": predicted, "recall": len(found) / len(gold),
                        # Precision@k uses requested k: a short result list
                        # does not receive artificially perfect precision.
                        "precision_at_k": len(found) / k, "complete": int(found == gold),
                        "reciprocal_rank": 1 / first_rank if first_rank else 0,
                        "context_document_ids": context_ids, "context_recall": len(context_found) / len(gold),
                        "context_precision": len(context_found) / len(context_ids) if context_ids else 0,
                        "context_complete": int(context_found == gold), "context_table_count": len(context_ids),
                        # Character count is a transparent size proxy, not a
                        # generator-token count; no generator is run here yet.
                        "context_char_count": sum(len(r["schema_text"]) for r in response["results"]),
                        "multi_table": len(gold) > 1, "elapsed_ms": response["elapsed_ms"]})
    def summarize(rows):
        if not rows:
            return {"questions": 0}
        timings = sorted(row["elapsed_ms"] for row in rows)
        return {"questions": len(rows), **{metric: statistics.mean(row[metric] for row in rows)
                for metric in ("recall", "precision_at_k", "complete", "reciprocal_rank", "context_recall",
                               "context_precision", "context_complete", "context_table_count", "context_char_count")},
                "p50_ms": statistics.median(timings), "p95_ms": timings[math.ceil(.95 * len(timings)) - 1]}
    return {**index.configuration(), "k": k, "scope": "database" if scoped else "global",
            "source_sha256": index.source_sha256,
            "labels_sha256": sha256(labels),
            "all": summarize(records), "multi_table": summarize([r for r in records if r["multi_table"]]),
            "latency_scope": "in-process retrieval only; excludes startup/network/generation",
            "records": records}

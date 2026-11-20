"""Build, search and evaluate schema retrieval and its optional refinements."""
import argparse
import json
from pathlib import Path
import time
from rag.bm25 import BM25Index
from rag.common.catalog import read_catalog
from rag.common.evaluation import evaluate

BACKENDS = ("bm25", "dense", "hybrid", "hybrid_graph", "hybrid_columns", "hybrid_rerank")
FOUR_RESULT_BACKENDS = {"hybrid_graph", "hybrid_columns", "hybrid_rerank"}


def index_class_for(backend: str):
    """Load only the selected backend; BM25 never imports embedding models."""
    if backend == "bm25":
        return BM25Index
    if backend == "dense":
        from rag.dense import DenseIndex
        return DenseIndex
    if backend == "hybrid":
        from rag.hybrid import HybridIndex
        return HybridIndex
    if backend == "hybrid_graph":
        from rag.hybrid_graph import HybridGraphIndex
        return HybridGraphIndex
    if backend == "hybrid_columns":
        from rag.hybrid_columns import HybridColumnsIndex
        return HybridColumnsIndex
    if backend == "hybrid_rerank":
        from rag.hybrid_rerank import HybridRerankIndex
        return HybridRerankIndex
    raise ValueError(f"Unknown retrieval backend: {backend}")


def main(default_backend: str = "bm25"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Completed RAG bundle directory")
    parser.add_argument("--index", type=Path, required=True, help="Versioned JSON schema index")
    parser.add_argument("--backend", choices=BACKENDS, default=default_backend)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build schema-only index; no model training")
    build.add_argument("--k1", type=float, default=1.2)
    build.add_argument("--b", type=float, default=.75)
    build.add_argument("--batch-size", type=int, default=32, help="Dense encoder batch size")
    build.add_argument("--rrf-constant", type=int, default=60, help="Hybrid rank-fusion constant")
    build.add_argument("--graph-alpha", type=float, default=.5, help="Question influence on FK path costs, 0<alpha<1")
    build.add_argument("--max-added-tables", type=int, default=10, help="Graph schema budget beyond k seeds")
    build.add_argument("--columns-per-table", type=int, default=3, help="Semantic columns per table, plus required keys")
    build.add_argument("--candidate-pool", type=int, default=8, help="Hybrid table candidates for cross-encoder reranking")
    parser.add_argument("--device", default="cpu", help="Dense encoder device, e.g. cpu or cuda")
    search = commands.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--db-id", help="Optional application-selected database")
    search.add_argument("--explain", action="store_true")
    search.add_argument("--k", type=int, help="Results/seeds: default 4 for graph/columns/rerank, 5 otherwise")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--scoped", action="store_true", help="Supply known database; easier than routing")
    evaluation.add_argument("--k", type=int, help="Results/seeds: default 4 for graph/columns/rerank, 5 otherwise")
    evaluation.add_argument("--output", type=Path, required=True, help="New report JSON file")
    args = parser.parse_args()
    if args.command != "build" and args.k is None:
        args.k = 4 if args.backend in FOUR_RESULT_BACKENDS else 5
    try:
        index_class = index_class_for(args.backend)
        if args.command == "build":
            started = time.perf_counter()
            docs, checksum = read_catalog(args.bundle)
            if args.backend == "bm25":
                index = BM25Index(docs, checksum, k1=args.k1, b=args.b)
            else:
                from rag.dense import DenseIndex
                from rag.dense.encoder import MiniLMEncoder
                index = DenseIndex(docs, MiniLMEncoder(device=args.device), checksum, batch_size=args.batch_size)
                if args.backend != "dense":
                    from rag.hybrid import HybridIndex
                    index = HybridIndex(BM25Index(docs, checksum, k1=args.k1, b=args.b), index,
                                        rrf_constant=args.rrf_constant)
                    if args.backend == "hybrid_graph":
                        index = index_class(index, alpha=args.graph_alpha,
                                            max_added_tables=args.max_added_tables, batch_size=args.batch_size)
                    elif args.backend == "hybrid_columns":
                        index = index_class(index, columns_per_table=args.columns_per_table,
                                            batch_size=args.batch_size)
                    elif args.backend == "hybrid_rerank":
                        from rag.hybrid_rerank.model import MiniLMCrossEncoder
                        index = index_class(index, MiniLMCrossEncoder(device=args.device),
                                            candidate_pool=args.candidate_pool)
            index.save(args.index)
            result = {"index": str(args.index), "tables": len(docs), "source_sha256": checksum,
                      "build_ms": (time.perf_counter() - started) * 1000}
        else:
            index = (index_class.load(args.index, args.bundle, device=args.device)
                     if args.backend != "bm25" else index_class.load(args.index, args.bundle))
            if args.command == "search":
                result = index.search(args.query, k=args.k, db_id=args.db_id, explain=args.explain)
            else:
                result = evaluate(index, args.bundle / f"labels/{args.split}.jsonl", k=args.k, scoped=args.scoped)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as file:
                    json.dump(result, file, indent=2)
                result = {key: value for key, value in result.items() if key != "records"}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"RAG error: {error}\n")


if __name__ == "__main__":
    main()

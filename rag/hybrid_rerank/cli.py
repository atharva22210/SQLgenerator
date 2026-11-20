"""Run with python -m rag.hybrid_rerank.cli."""
from rag.common.cli import main

if __name__ == "__main__":
    main(default_backend="hybrid_rerank")

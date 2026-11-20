"""Run with python -m rag.bm25.cli."""
from rag.common.cli import main

if __name__ == "__main__":
    main(default_backend="bm25")

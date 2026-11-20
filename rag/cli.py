"""Compatibility entry point; BM25 remains the default backend."""
from rag.common.cli import main
from rag.common.evaluation import evaluate

if __name__ == "__main__":
    main()

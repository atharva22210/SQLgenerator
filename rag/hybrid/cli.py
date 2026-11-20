"""Run with python -m rag.hybrid.cli; uses the shared CLI implementation."""
from rag.common.cli import main

if __name__ == "__main__":
    main(default_backend="hybrid")

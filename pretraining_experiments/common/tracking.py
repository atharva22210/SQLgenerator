"""Compatibility path for training.tracking."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("training.tracking", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("training.tracking")

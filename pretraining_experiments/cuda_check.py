"""Compatibility path for pretraining_experiments.sequence_packing.check."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("pretraining_experiments.sequence_packing.check", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("pretraining_experiments.sequence_packing.check")

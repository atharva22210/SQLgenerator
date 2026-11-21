"""Compatibility import; implementation lives in pretraining_experiments.cuda_check."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("pretraining_experiments.cuda_check", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("pretraining_experiments.cuda_check")

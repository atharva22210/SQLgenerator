"""Small exact FAISS backend; isolate its OpenMP runtime on macOS.

macOS PyTorch and FAISS wheels bundle separate libomp copies. A persistent
worker keeps FAISS out of the encoder process without disabling runtime checks.
The worker runs this file directly, importing neither our encoder nor torch.
"""
import json
from pathlib import Path
import subprocess
import sys
import threading
import weakref

import numpy as np


class _FlatIndex:
    """FAISS operations shared by the direct backend and isolated worker."""
    def __init__(self, vectors):
        import faiss

        vectors = np.asarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, candidates):
        import faiss

        # Filter global document positions inside FAISS, before ranking.
        selector = faiss.IDSelectorBatch(np.asarray(candidates, dtype=np.int64))
        parameters = faiss.SearchParameters(sel=selector)
        scores, positions = self.index.search(np.asarray(query, dtype=np.float32), len(candidates),
                                             params=parameters)
        return scores[0], positions[0]


def _close_worker(process):
    """Release pipes and stop a worker when its owning index is collected."""
    process.stdin.close()
    if process.poll() is None:
        process.terminate()
    process.wait()
    process.stdout.close()


class FaissIndex:
    """Store normalized float32 [N,H] vectors once; search permitted row IDs.

    macOS uses a persistent subprocess; other platforms call FAISS directly.
    search returns (scores, document positions) for every eligible vector so
    DenseIndex can preserve deterministic ties at the final top-k boundary.
    """
    def __init__(self, vectors: np.ndarray):
        self.ntotal = len(vectors)
        self._lock = threading.Lock()
        self._local = None
        if sys.platform != "darwin":
            self._local = _FlatIndex(vectors)
        else:
            self._worker = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            self._cleanup = weakref.finalize(self, _close_worker, self._worker)
            try:
                self._exchange({"vectors": vectors.tolist()})
            except Exception:
                self._cleanup()
                raise

    def _exchange(self, payload):
        self._worker.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self._worker.stdin.flush()
        response = self._worker.stdout.readline()
        if not response:
            raise OSError("FAISS worker stopped; inspect its stderr for the cause")
        return json.loads(response)

    def search(self, query: np.ndarray, candidates: list[int]):
        """Score normalized [1,H] query against nonempty permitted document IDs."""
        if self._local is not None:
            return self._local.search(query, candidates)
        # Serialize each request/reply pair if the index is shared by callers.
        with self._lock:
            result = self._exchange({"query": query.tolist(), "candidates": candidates})
        return np.asarray(result["scores"], dtype=np.float32), np.asarray(result["positions"], dtype=np.int64)


def _worker_main():
    initial = json.loads(sys.stdin.readline())
    index = _FlatIndex(initial["vectors"])
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        scores, positions = index.search(request["query"], request["candidates"])
        print(json.dumps({"scores": scores.tolist(), "positions": positions.tolist()}), flush=True)


if __name__ == "__main__":
    _worker_main()

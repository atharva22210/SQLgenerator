import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.byte_bpe.train_byte_bpe import documents


class BPEShardTests(unittest.TestCase):
    def test_shards_preserve_empty_and_cross_boundary_documents(self):
        # Test every shard width, including boundaries on EOS and inside a row.
        rows = [[11, 12, 13], [], [21], [31, 32, 33, 34]]
        ids = [token for row in rows for token in [*row, 99]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.bin"
            np.asarray(ids, dtype="<u4").tofile(path)
            for width in range(1, len(ids) + 1):
                actual = [row for start in range(0, len(ids), width)
                          for batch in documents(path, 99, start, min(start+width, len(ids)))
                          for row in batch]
                self.assertEqual(actual, rows)

"""Focused checks for resumable token files; no Modal job is launched."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data import prepare_pretrain_data as job


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.fail_on = None

    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == self.fail_on:
            raise RuntimeError("simulated interruption")
        return [int(text)]


class FakeVolume:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


class CheckpointTests(unittest.TestCase):
    def test_resume_truncates_uncheckpointed_bytes_without_duplicating_rows(self):
        # Source row zero belongs to validation; rows one through four are
        # the train partition that this test checkpoints and resumes.
        rows = [{"text": str(value)} for value in range(5)]
        tokenizer = FakeTokenizer()
        tokenizer.fail_on = "3"
        volume = FakeVolume()
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            with (
                patch.object(job, "OUTPUT_PATH", output_path),
                patch.object(job, "volume", volume),
                patch.object(job, "CHECKPOINT_EVERY_ROWS", 2),
                patch.object(job, "load_stream", return_value=rows),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    job.write_split("sample.bin", "test", lambda row: row["text"], tokenizer, "train")
                checkpoint_path = output_path / "sample.bin.checkpoint.json"
                self.assertEqual(json.loads(checkpoint_path.read_text())["rows_done"], 2)

                # Simulate a crash after extra data reached the Volume but
                # before its row count was checkpointed.
                with (output_path / "sample.bin").open("ab") as output:
                    np.asarray([123], dtype="uint32").tofile(output)
                tokenizer.fail_on = None
                count = job.write_split("sample.bin", "test", lambda row: row["text"], tokenizer, "train")

                self.assertEqual(count, 8)
                self.assertEqual(np.fromfile(output_path / "sample.bin", dtype="uint32").tolist(),
                                 [1, 99, 2, 99, 3, 99, 4, 99])
                self.assertTrue(json.loads(checkpoint_path.read_text())["complete"])
                self.assertEqual(volume.commits, 3)


if __name__ == "__main__":
    unittest.main()

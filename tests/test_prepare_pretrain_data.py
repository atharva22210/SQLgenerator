"""Offline correctness checks: no Modal runs, downloads, or performance claims."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from data import prepare_pretrain_data as job


class FakeTokenizer:
    """Keep token IDs readable, with an injectable batch failure."""

    eos_token_id = 99
    special_tokens_map = {"eos_token": "<eos>"}
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"vocab": "test"}')

    def __init__(self):
        self.fail_on = None
        self.batches = []

    def __len__(self):
        return 100

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, padding=False, truncation=False,
                              return_attention_mask=False, return_token_type_ids=False)
        self.batches.append(texts)
        if self.fail_on in texts:
            raise RuntimeError("simulated interruption")
        return {"input_ids": [[int(text)] for text in texts]}


class FakeStream:
    """Track visited source rows to detect full replay and prefetch mistakes."""

    def __init__(self, rows):
        self.rows = rows
        self.position = 0
        self.visited = []
        self.restored = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.rows):
            raise StopIteration
        self.visited.append(self.position)
        row = self.rows[self.position]
        self.position += 1
        return row

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]
        self.restored = state["position"]


def format_number(row):
    return row["text"]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.rows = [{"text": str(i)} for i in range(31)]
        self.tokenizer = FakeTokenizer()
        self.real_loader = job.load_stream
        self.streams = []
        self.volume = SimpleNamespace(commit=lambda: None)
        for key, value in dict(OUTPUT_PATH=self.output, volume=self.volume, BATCH_ROWS=2,
                               CHECKPOINT_EVERY_ROWS=4, CHECKPOINT_SECONDS=100000).items():
            self.stack.enter_context(patch.object(job, key, value))
        self.load = self.stack.enter_context(patch.object(job, "load_stream", side_effect=self.new_stream))
        self.info = self.stack.enter_context(patch("huggingface_hub.HfApi.dataset_info",
                                                  return_value=SimpleNamespace(sha="pinned-sha")))

    def new_stream(self, *args, **kwargs):
        stream = FakeStream(self.rows)
        self.streams.append(stream)
        return stream

    def run_job(self, formatter=format_number):
        return job.write_corpus("sample", "test", formatter, self.tokenizer)

    def state(self):
        return json.loads((self.output / "sample.checkpoint.json").read_text())

    def assert_reference(self, formatter=format_number, tokenizer=None):
        for split in ("train", "val"):
            expected = []
            for i, row in enumerate(self.rows):
                text = formatter(row)
                if (i % 10 == 0) == (split == "val") and text is not None:
                    expected.extend(tokenizer.encode(text, add_special_tokens=False)
                                    if tokenizer else [int(text)])
                    expected.append(99)
            actual = (self.output / f"sample_{split}.bin").read_bytes()
            self.assertEqual(actual, np.asarray(expected, dtype="uint32").tobytes())

    def test_single_pass_matches_original_bytes_and_batches(self):
        self.run_job()
        self.assert_reference()
        self.load.assert_called_once_with("test", config=None, revision="pinned-sha")
        self.assertTrue(any(len(batch) > 1 for batch in self.tokenizer.batches))
        self.assertEqual(self.state()["source_rows"], 31)
        self.assertTrue(all(s["complete"] for s in self.state()["splits"].values()))

    def test_resume_uses_saved_cursor_despite_prefetch_and_truncates_both_files(self):
        self.tokenizer.fail_on = "6"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job()
        self.assertEqual(self.state()["source_rows"], 4)
        self.assertGreater(self.streams[0].position, 4)  # speculative read-ahead
        for split in ("train", "val"):
            with (self.output / f"sample_{split}.bin").open("ab") as output:
                output.write(b"uncheckpointed bytes")
        self.tokenizer.fail_on = None
        self.run_job()
        self.assertEqual(self.streams[1].restored, 4)
        self.assertEqual(self.streams[1].visited[0], 4)
        self.assert_reference()
        # A restart uses the saved revision even if Hub main has changed.
        self.info.assert_called_once()

    def test_split_index_and_skipped_rows_survive_non_multiple_of_ten_checkpoint(self):
        def formatter(row):
            return None if row["text"] in {"7", "20"} else row["text"]
        self.stack.enter_context(patch.object(job, "CHECKPOINT_EVERY_ROWS", 6))
        self.tokenizer.fail_on = "8"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job(formatter)
        self.assertEqual(self.state()["source_rows"], 6)
        self.tokenizer.fail_on = None
        self.run_job(formatter)
        self.assert_reference(formatter)

    def test_failure_before_first_periodic_checkpoint_can_resume(self):
        self.tokenizer.fail_on = "2"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job()
        self.assertEqual(self.state()["source_rows"], 0)
        self.assertIsNone(self.state()["stream_state"])
        self.tokenizer.fail_on = None
        self.run_job()
        self.assert_reference()

    def test_source_timeout_propagates_instead_of_looping_on_heartbeat(self):
        self.load.side_effect = None
        self.load.return_value = iter([])
        with patch.object(job, "read_batch", side_effect=TimeoutError("source timeout")):
            with self.assertRaisesRegex(TimeoutError, "source timeout"):
                self.run_job()

    def test_legacy_validation_prefix_is_preserved_while_missing_train_is_filled(self):
        state = dict(dataset="test", config=None, split="val", dtype="uint32", vocab_size=100,
                     rows_done=2, tokens_written=4, complete=False)
        (self.output / "sample_val.bin.checkpoint.json").write_text(json.dumps(state))
        np.asarray([0, 99, 10, 99], dtype="uint32").tofile(self.output / "sample_val.bin")
        self.tokenizer.fail_on = "6"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job()
        self.assertEqual(self.state()["splits"]["val"]["rows_done"], 2)
        self.tokenizer.fail_on = None
        self.run_job()
        self.assert_reference()
        flattened = [text for batch in self.tokenizer.batches for text in batch]
        self.assertNotIn("0", flattened)
        self.assertNotIn("10", flattened)

    def test_changed_tokenizer_is_rejected_before_files_are_mutated(self):
        self.run_job()
        original = (self.output / "sample_train.bin").read_bytes()
        self.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"vocab": "changed"}')
        with self.assertRaisesRegex(ValueError, "different input data or code"):
            self.run_job()
        self.assertEqual((self.output / "sample_train.bin").read_bytes(), original)

    def test_short_output_is_rejected_before_other_file_is_truncated(self):
        self.run_job()
        train = self.output / "sample_train.bin"
        train.write_bytes(train.read_bytes() + b"tail")
        (self.output / "sample_val.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "shorter than its checkpoint"):
            self.run_job()
        self.assertTrue(train.read_bytes().endswith(b"tail"))

    def test_orphaned_output_is_not_overwritten(self):
        path = self.output / "sample_val.bin"
        path.write_bytes(b"unowned")
        with self.assertRaisesRegex(ValueError, "no checkpoint"):
            self.run_job()
        self.assertEqual(path.read_bytes(), b"unowned")

    def test_completed_corpus_requires_no_stream_or_hub_lookup(self):
        first = self.run_job()
        self.load.reset_mock()
        self.info.reset_mock()
        self.assertEqual(self.run_job(), first)
        self.load.assert_not_called()
        self.info.assert_not_called()

    def test_character_budget_keeps_large_documents_intact(self):
        stream = FakeStream([{"text": "long document"}, {"text": "next"}])
        floors = {split: {"rows_done": 0, "complete": False} for split in ("train", "val")}
        with patch.object(job, "BATCH_CHARACTERS", 3):
            batch = job.read_batch(iter(stream), stream, 0, format_number, floors)
        self.assertEqual(batch["texts"], ["long document"])
        self.assertEqual(batch["source_rows"], 1)

    def test_time_limit_checkpoints_before_row_threshold(self):
        self.stack.enter_context(patch.object(job, "CHECKPOINT_EVERY_ROWS", 100000))
        self.stack.enter_context(patch.object(job, "CHECKPOINT_SECONDS", 0))
        self.tokenizer.fail_on = "4"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job()
        self.assertEqual(self.state()["source_rows"], 4)

    def test_fineweb_loader_projects_text_and_uses_ordered_row_groups(self):
        # Bypass setUp's I/O stub so the production loader itself is checked.
        stream = MagicMock()
        with patch("datasets.load_dataset", return_value=stream) as loader:
            result = self.real_loader(job.ENGLISH_DATASET, config=job.ENGLISH_CONFIG,
                                      revision="fixed-revision")
        loader.assert_called_once_with(path=job.ENGLISH_DATASET, split="train", streaming=True,
                                       revision="fixed-revision", columns=["text"],
                                       name=job.ENGLISH_CONFIG)
        stream.reshard.assert_called_once_with()
        self.assertIs(result, stream.reshard.return_value)

    def test_real_fast_tokenizer_matches_per_document_encode(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(models.WordLevel({"<unk>": 98, "<eos>": 99,
                                              **{str(i): i for i in range(98)}} , unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        real = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")
        self.rows = [{"text": text} for text in ["1 2", "3", "4 5 6", "Unicode café", "7 " * 2000]]
        self.tokenizer = real
        self.run_job()
        self.assert_reference(tokenizer=real)

    def test_real_parquet_stream_restores_across_shards(self):
        from datasets import load_dataset
        import pyarrow as pa
        import pyarrow.parquet as pq
        parquet_paths = []
        for shard in range(3):
            path = self.output / f"source-{shard}.parquet"
            pq.write_table(pa.Table.from_pylist(self.rows[shard * 10:(shard + 1) * 10]),
                           path, row_group_size=3)
            parquet_paths.append(str(path))
        self.rows = self.rows[:30]
        self.load.side_effect = lambda *args, **kwargs: load_dataset(
            "parquet", data_files=parquet_paths, split="train", streaming=True, batch_size=2,
            cache_dir=str(self.output / "hf-cache")).reshard()
        self.tokenizer.fail_on = "16"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_job()
        state = self.state()
        self.assertEqual(state["source_rows"], 16)
        self.tokenizer.fail_on = None
        self.run_job()
        self.assert_reference()


class SubmissionTests(unittest.TestCase):
    def test_submit_only_uses_independent_spawn_without_waiting(self):
        with patch.object(job, "prepare_pretrain_data") as function, patch("builtins.print"):
            job.main(submit_only=True)
        function.spawn.assert_called_once_with()
        function.spawn.return_value.get.assert_not_called()
        function.remote.assert_not_called()

    def test_waiting_mode_still_uses_independent_spawn(self):
        with patch.object(job, "prepare_pretrain_data") as function, patch("builtins.print"):
            function.spawn.return_value.get.return_value = {"complete": True}
            job.main()
        function.spawn.assert_called_once_with()
        function.spawn.return_value.get.assert_called_once_with()
        function.remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()

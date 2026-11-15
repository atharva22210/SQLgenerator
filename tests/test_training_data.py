"""Check document boundaries and schedule before spending time on a GPU."""

import random
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from data.dataloader import DocumentFile, SQL_SOURCES, make_batch
from train_pretrain import TrainConfig, learning_rate, _validate_corpus_files


class FixedCorpus:
    """Supply two short documents so their boundary is easy to inspect."""

    def __init__(self) -> None:
        self.calls = []
        self.first_slices = []

    def sample(self, kind, rng, length, *, first_slice=False):
        self.calls.append(kind)
        self.first_slices.append(first_slice)
        return ([1, 2, 9] if len(self.calls) % 2 else [3, 4, 9])[:length]


class TrainingDataTests(unittest.TestCase):
    def test_preparation_validation_supports_migrated_and_legacy_markers(self) -> None:
        """Migration keeps stale split markers; only the v2 marker advances."""
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            metadata = {"token_dtype": "uint32", "tokenizer_vocab_size": 10,
                        "pipeline_version": 2, "tokenizer_sha256": "test-hash",
                        "dataset_revisions": {}, "token_counts": {}}
            for name in ("english", *SQL_SOURCES):
                for split in ("train", "val"):
                    key = f"{name}_{split}"
                    np.array([1, 9], dtype="<u4").tofile(directory / f"{key}.bin")
                    metadata["token_counts"][key] = 2
                    (directory / f"{key}.bin.checkpoint.json").write_text(json.dumps(
                        {"tokens_written": 2, "complete": True}))
            _validate_corpus_files(directory, metadata)
            old = directory / "english_val.bin.checkpoint.json"
            old.write_text(json.dumps({"tokens_written": 1, "complete": False}))
            with self.assertRaisesRegex(ValueError, "unfinished"):
                _validate_corpus_files(directory, metadata)
            source = {"pipeline_version": 2, "dtype": "uint32", "vocab_size": 10,
                      "tokenizer_sha256": "test-hash", "revision": None,
                      "splits": {split: {"tokens_written": 2, "complete": True}
                                 for split in ("train", "val")}}
            current = directory / "english.checkpoint.json"
            current.write_text(json.dumps(source))
            _validate_corpus_files(directory, metadata)
            source["splits"]["val"]["complete"] = False
            current.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "unfinished"):
                _validate_corpus_files(directory, metadata)
            source["splits"]["val"] = {"tokens_written": 1, "complete": True}
            current.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "token count"):
                _validate_corpus_files(directory, metadata)
            source["tokenizer_sha256"] = "different-tokenizer"
            current.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _validate_corpus_files(directory, metadata)

    def test_uint32_file_recovers_complete_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.bin"
            np.array([1, 2, 9, 3, 4, 9], dtype="<u4").tofile(path)
            source = DocumentFile(path, eos_id=9)
            self.assertEqual(source.ends.tolist(), [2, 5])
            for seed in range(10):
                self.assertIn(source.sample(random.Random(seed), 6), ([1, 2, 9], [3, 4, 9]))

    def test_packing_masks_cross_document_target_and_attention(self) -> None:
        masked = make_batch(
            FixedCorpus(), mode="masked_pack", batch_size=1, seq_len=6,
            sql_fraction=0.5, pad_id=0, rng=random.Random(0),
        )
        self.assertEqual(masked.input_ids.tolist(), [[1, 2, 9, 3, 4, 9]])
        self.assertEqual(masked.labels.tolist(), [[1, 2, 9, -100, 4, 9]])
        self.assertEqual(masked.position_ids.tolist(), [[0, 1, 2, 0, 1, 2]])
        inputs = masked.model_inputs("masked_pack", torch.device("cpu"))
        self.assertNotIn("attention_mask", inputs)
        layout = inputs["attention_layout"]
        self.assertEqual(layout.cu_seqlens.tolist(), [0, 3, 6])
        self.assertEqual(layout.cu_seqlens.dtype, torch.int32)
        self.assertEqual(layout.max_seqlen, 3)
        attention = layout.dense_mask(1, 6)
        self.assertFalse(bool(attention[0, 3, 2]))
        self.assertTrue(bool(attention[0, 3, 3]))

        naive = make_batch(
            FixedCorpus(), mode="naive_pack", batch_size=1, seq_len=6,
            sql_fraction=0.5, pad_id=0, rng=random.Random(0),
        )
        self.assertEqual(naive.labels[0, 3].item(), 3)
        self.assertEqual(naive.position_ids.tolist(), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(naive.attention_layout.cu_seqlens.tolist(), [0, 6])
        self.assertTrue(bool(naive.attention_layout.dense_mask(1, 6)[0, 3, 2]))

    def test_training_uses_document_prefixes_in_every_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long.bin"
            np.array([1, 2, 3, 4, 5, 6, 9], dtype="<u4").tofile(path)
            source = DocumentFile(path, eos_id=9)
            for seed in range(5):
                self.assertEqual(source.sample(random.Random(seed), 3), [1, 2, 3])
        for mode in ("pad", "naive_pack", "masked_pack"):
            corpus = FixedCorpus()
            make_batch(corpus, mode=mode, batch_size=2, seq_len=7,
                       sql_fraction=0.5, pad_id=0, rng=random.Random(0))
            self.assertTrue(all(corpus.first_slices))

    def test_layout_skips_padding_and_keeps_rows_independent(self) -> None:
        for mode, lengths in (("pad", (3, 3)), ("naive_pack", (7, 7)),
                              ("masked_pack", (3, 3, 1, 3, 3, 1))):
            with self.subTest(mode=mode):
                batch = make_batch(FixedCorpus(), mode=mode, batch_size=2, seq_len=7,
                                   sql_fraction=0.5, pad_id=0, rng=random.Random(0))
                layout = batch.attention_layout
                expected_indices = [0, 1, 2, 7, 8, 9] if mode == "pad" else list(range(14))
                self.assertEqual(layout.token_indices.tolist(), expected_indices)
                self.assertEqual(layout.sequence_lengths, lengths)
                self.assertEqual(layout.cu_seqlens[-1], batch.non_pad_tokens)
                self.assertEqual(layout.max_seqlen, max(lengths))
                self.assertEqual(batch.predicted_tokens, int(batch.labels[:, 1:].ne(-100).sum()))
                with self.assertRaisesRegex(ValueError, "match"):
                    batch.model_inputs("invalid", torch.device("cpu"))

    def test_all_ignored_targets_are_rejected_before_gpu_transfer(self) -> None:
        class OneTokenCorpus:
            def sample(self, kind, rng, length, *, first_slice):
                return [9]

        with self.assertRaisesRegex(ValueError, "no next-token targets"):
            make_batch(OneTokenCorpus(), mode="masked_pack", batch_size=2, seq_len=4,
                       sql_fraction=0.5, pad_id=0, rng=random.Random(0))

    def test_sql_mix_endpoints_and_padding(self) -> None:
        sql_corpus = FixedCorpus()
        sql = make_batch(sql_corpus, mode="pad", batch_size=2, seq_len=6,
                         sql_fraction=1.0, pad_id=0, rng=random.Random(0))
        self.assertEqual(sql_corpus.calls, ["sql", "sql"])
        self.assertEqual(sql.sql_tokens, 6)
        self.assertEqual(sql.labels[0].tolist(), [1, 2, 9, -100, -100, -100])
        self.assertEqual(sql.non_pad_tokens, 6)
        english_corpus = FixedCorpus()
        make_batch(english_corpus, mode="pad", batch_size=2, seq_len=6,
                   sql_fraction=0.0, pad_id=0, rng=random.Random(0))
        self.assertEqual(english_corpus.calls, ["english", "english"])

    def test_cosine_schedule_reaches_peak_then_minimum(self) -> None:
        config = TrainConfig(max_steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate(1, config), config.peak_lr / 2)
        self.assertAlmostEqual(learning_rate(2, config), config.peak_lr)
        self.assertAlmostEqual(learning_rate(10, config), config.min_lr)

    def test_cosine_without_warmup_starts_at_peak(self) -> None:
        config = TrainConfig(max_steps=4, warmup_steps=0)
        self.assertAlmostEqual(learning_rate(1, config), config.peak_lr)
        self.assertAlmostEqual(learning_rate(4, config), config.min_lr)

    def test_inverse_sqrt_schedule_anchors_at_warmup_and_decays(self) -> None:
        config = TrainConfig(lr_schedule="inverse_sqrt", max_steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate(1, config), config.peak_lr / 2)
        self.assertAlmostEqual(learning_rate(2, config), config.peak_lr)
        # Four times as many updates after the anchor halves the rate.
        self.assertAlmostEqual(learning_rate(8, config), config.peak_lr / 2)
        self.assertGreater(learning_rate(10, config), config.min_lr)

    def test_inverse_sqrt_without_warmup_and_invalid_schedule(self) -> None:
        config = TrainConfig(lr_schedule="inverse_sqrt", max_steps=4, warmup_steps=0)
        self.assertAlmostEqual(learning_rate(1, config), config.peak_lr)
        self.assertAlmostEqual(learning_rate(4, config), config.peak_lr / 2)
        with self.assertRaisesRegex(ValueError, "lr_schedule"):
            TrainConfig(lr_schedule="linear")


if __name__ == "__main__":
    unittest.main()

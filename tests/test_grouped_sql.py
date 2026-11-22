from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from data.grouped_corpus import GROUPED_SQL_FORMAT_VERSION, resolve_grouped_sql
from data.grouped_sql import (SQLFingerprintTokenizer,
                              repository_sql_text, split_and_audit, stage_rows, statement_hashes)
from data.prepare_grouped_sql import split_examples, write_split


class GroupedSQLTests(unittest.TestCase):
    def test_lexical_hash_preserves_literals_and_normalizes_formatting(self):
        lexer = SQLFingerprintTokenizer()
        original = statement_hashes("SELECT 'A;B' FROM t; -- comment", lexer)
        self.assertEqual(original, statement_hashes("select 'A;B' from T", lexer))
        self.assertNotEqual(original, statement_hashes("select 'a;b' from t", lexer))
        self.assertEqual(len(original), 1)
        self.assertEqual(len(statement_hashes("SELECT `A;B` FROM t;", lexer)), 1)

    def test_only_leading_starcoder_headers_are_removed(self):
        body = "SELECT '<filename>literal.sql';"
        content = "\ufeff<reponame>org/repo\r\n<filename>test.sql<gh_stars>0\r\n" + body
        self.assertEqual(repository_sql_text(content), body)
        self.assertEqual(repository_sql_text(body), body)

    def test_dedup_and_reject_query_copied_to_held_out_repo(self):
        with closing(sqlite3.connect(":memory:")) as db:
            syn = [dict(db_id=f"db{i}", schema="CREATE TABLE example (id INT);",
                        question=f"Question {i}", sql=f"SELECT {i};") for i in range(10)]
            syn[0]["external_knowledge"] = "Required domain definition"
            syn.append({**syn[0], "sql": "select 0 -- same answer"})
            repos = [
                dict(max_stars_repo_name=f"example/repo{i}", content=f"SELECT {100 + i};")
                for i in range(10)
            ]
            # Discover the deterministic held-out groups before adding the
            # cross-source copy. Both groups already have a clean document.
            stage_rows(db, syn, repos)
            split_and_audit(db, seed=42)
            held_repo, = db.execute("SELECT group_id FROM documents WHERE kind='repo_name' AND split='val'").fetchone()
            train_query, = db.execute("SELECT text FROM documents WHERE kind='db_id' AND split='train' LIMIT 1").fetchone()
            copied = train_query.split("<sql>\n")[1].split("\n</sql>")[0]

        with closing(sqlite3.connect(":memory:")) as db:
            repos.append(dict(max_stars_repo_name=held_repo.upper(), content=f"<filename>copied.sql<gh_stars>20\n{copied}\nSELECT 999;"))
            staged = stage_rows(db, syn, repos)
            audit = split_and_audit(db, seed=42)
            self.assertEqual(staged["db_id"]["duplicates"], 1)
            document, = db.execute("SELECT text FROM documents WHERE group_id = 'db0'").fetchone()
            self.assertIn("Additional context:\nRequired domain definition", document)
            self.assertEqual(audit["assigned_groups"], {
                "db_id": {"train": 9, "val": 1}, "repo_name": {"train": 9, "val": 1},
            })
            self.assertEqual(audit["overlaps"], dict(db_id=0, repo_name=0, query_hash=0))
            self.assertEqual(audit["dropped_validation_documents"], 1)
            self.assertEqual(audit["retained"]["repo_name/val"]["documents"], 1)
            self.assertEqual({example["kind"] for example in split_examples(db, "val")},
                             {"db_id", "repo_name"})
            self.assertFalse(any("SELECT 999" in example["text"]
                                 for example in split_examples(db, "val")))

            class FakeTokenizer:
                eos_token_id = 99

                def __call__(self, texts, **kwargs):
                    return {"input_ids": [[8, 9] for _ in texts]}

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sql_val.bin"
                self.assertEqual(write_split(db, "val", path, FakeTokenizer()), 6)
                self.assertEqual(np.fromfile(path, dtype="<u4").tolist(), [8, 9, 99, 8, 9, 99])


    def test_parallel_fingerprinting_preserves_source_order_and_split(self):
        syn = [dict(db_id=f"db{i}", schema="CREATE TABLE t (id INT)",
                    question=f"Question {i}", sql=f"SELECT {i};") for i in range(80)]
        # Match a query from a different group in an earlier worker batch.
        syn.append({**syn[0], "db_id": "late_duplicate", "sql": "select 0"})
        repos = [dict(max_stars_repo_name=f"org/repo{i}", content=f"SELECT {100 + i};")
                 for i in range(80)]
        snapshots = []
        for workers in (1, 4):
            with closing(sqlite3.connect(":memory:")) as db:
                staging = stage_rows(db, syn, repos, workers=workers)
                audit = split_and_audit(db, seed=42)
                documents = db.execute(
                    "SELECT kind, group_id, digest, split FROM documents ORDER BY id"
                ).fetchall()
                snapshots.append((staging, audit, documents))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1][0]["db_id"]["duplicates"], 1)

    def test_worker_loads_audited_overlay_without_legacy_sql_files(self):
        from distributed_training import sql_validation
        from training import config as training_config
        from data import prepare_pretrain_data
        from transformers import AutoTokenizer

        class FakeTokenizer:
            eos_token_id = 99
            pad_token_id = 0

            def __len__(self):
                return 100

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "grouped_sql" / "completed"
            overlay.mkdir(parents=True)
            identity = dict(tokenizer_vocab_size=100, eos_token_id=99, token_dtype="uint32")
            (root / "metadata.json").write_text(json.dumps({
                **identity, "token_counts": {"english_train": 2, "english_val": 2},
            }))
            metadata = {
                **identity, "format_version": GROUPED_SQL_FORMAT_VERSION, "tokenizer_sha256": "tokenizer",
                "token_counts": {"sql_train": 2, "sql_val": 2},
                "audit": {"overlaps": dict(db_id=0, repo_name=0, query_hash=0)},
            }
            (overlay / "metadata.json").write_text(json.dumps(metadata))
            for path, token in ((root / "english_train.bin", 1), (root / "english_val.bin", 2),
                                (overlay / "sql_train.bin", 3), (overlay / "sql_val.bin", 4)):
                np.asarray([token, 99], dtype="<u4").tofile(path)
            # An interrupted build may have published a preview only.
            unfinished = overlay.parent / "unfinished"
            unfinished.mkdir()
            (unfinished / "examples.json").write_text("{}")
            self.assertEqual(resolve_grouped_sql(root)[0], overlay)
            with (patch.object(training_config, "DATA_DIR", root),
                  patch.object(training_config.volume, "reload"),
                  patch.object(training_config, "_tokenizer_path", return_value=root),
                  patch.object(AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()),
                  patch.object(prepare_pretrain_data, "tokenizer_fingerprint", return_value="tokenizer")):
                _, train, validation = sql_validation.load_training_and_sql_validation()
                _, _, packing_validation = training_config._load_data_and_tokenizer()
                self.assertEqual(packing_validation.english.tokens.tolist(), [2, 99])
                self.assertEqual(packing_validation.sql[0].tokens.tolist(), [4, 99])
                self.assertEqual(train.sql[0].tokens.tolist(), [3, 99])
                self.assertEqual(validation.sql[0].tokens.tolist(), [4, 99])
                self.assertEqual(list(validation.sql[0].eligible_indices), [0])
                metadata["audit"]["overlaps"]["query_hash"] = 1
                (overlay / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "overlap audit"):
                    sql_validation.load_training_and_sql_validation(overlay)


    def test_resolver_rejects_preview_only_and_missing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "grouped_sql" / "build"
            build.mkdir(parents=True)
            (build / "examples.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "No completed Grouped SQL"):
                resolve_grouped_sql(root)
            metadata = dict(format_version=GROUPED_SQL_FORMAT_VERSION, token_dtype="uint32",
                            token_counts=dict(sql_train=2, sql_val=2),
                            audit={"overlaps": dict(db_id=0, repo_name=0, query_hash=0)})
            (build / "metadata.json").write_text(json.dumps(metadata))
            np.asarray([1, 99], dtype="<u4").tofile(build / "sql_train.bin")
            with self.assertRaisesRegex(ValueError, "sql_val.bin"):
                resolve_grouped_sql(root)

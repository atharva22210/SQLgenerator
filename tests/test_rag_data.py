"""Offline checks for schema integrity and evaluation leakage boundaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from rag import prepare_data as prepare


def row(index, *, question=None, context=None, stage="test", context_hash=None):
    """Tiny source fixture; no live service or credentials are needed."""
    context = context or "CREATE TABLE t (id INT PRIMARY KEY, name TEXT); INSERT INTO t VALUES (1, 'example');"
    return {"id": index, "source_index": index, "source_split": "test" if stage == "test" else "train",
            "dataset_revision": prepare.REVISION, "domain": "learning", "domain_description": "A sample domain",
            "sql_context": context, "sql_prompt": question or f"Find item {index}", "sql": "SELECT name FROM t",
            "context_sha256": context_hash or hashlib.sha256(context.encode()).hexdigest(),
            "db_name": f"dbpt012345abcdefx{'test' if stage == 'test' else 'train'}{index:06d}",
            "result_row_count": 1}


class SchemaTests(unittest.TestCase):
    def test_catalogue_contains_schema_not_source_answers_or_values(self):
        record = row(1, question="PRIVATE HELD OUT QUESTION")
        document = prepare.extract_schema(record)[0]
        text = prepare.render_document(document)
        self.assertEqual(document["primary_key"], ["id"])
        self.assertNotIn(record["sql_prompt"], text)
        self.assertNotIn(record["sql"], text)
        self.assertNotIn("example", text)
        self.assertNotIn("INSERT", document["schema_sql"])
        self.assertEqual(document["foreign_keys"], [])

    def test_composite_foreign_key_is_preserved_without_guessing_other_joins(self):
        context = ("CREATE TABLE parent (a INT, b INT, PRIMARY KEY (a,b));"
                   "CREATE TABLE child (a INT, b INT, other_id INT, "
                   "FOREIGN KEY (a,b) REFERENCES parent(a,b));")
        documents = prepare.extract_schema(row(1, context=context))
        self.assertEqual(documents[0]["primary_key"], ["a", "b"])
        self.assertEqual(documents[1]["foreign_keys"], [{"columns": ["a", "b"],
            "referenced_table": "parent", "referenced_columns": ["a", "b"]}])

    def test_gold_tables_exclude_cte_and_derived_aliases(self):
        query = ("WITH recent AS (SELECT * FROM orders) SELECT * FROM recent r "
                 "JOIN (SELECT * FROM customers) c ON r.customer_id=c.id")
        self.assertEqual(prepare.gold_tables(query), ["customers", "orders"])

    def test_nonretained_database_is_rejected(self):
        record = row(1)
        record["db_name"] = "sandbox_db"
        with self.assertRaises(ValueError):
            prepare.extract_schema(record)

    def test_live_metadata_mismatch_fails_before_writing(self):
        # An empty metadata result models databases that have been deleted or
        # permissions that can no longer see them. Never publish stale schemas.
        with patch("pymysql.connect") as connect, patch("infra.db_client.connection_options", return_value={}):
            cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            cursor.fetchone.return_value = {"version": "8.4.11"}
            cursor.fetchall.return_value = []
            with self.assertRaisesRegex(ValueError, "Live databases/tables differ"):
                prepare.snapshot_live_tables(prepare.extract_schema(row(1)))
            self.assertFalse(any("CREATE" in call.args[0] or "INSERT" in call.args[0]
                                 for call in cursor.execute.call_args_list))


class BundleTests(unittest.TestCase):
    def sources(self):
        # Different hashes place dev/test on opposite sides and keep the
        # fixture source contexts disjoint, as ingestion promises in reality.
        return {"sft": [row(10, question="Already seen", stage="sft", context_hash="1" * 64)],
                "rl": [row(20, stage="rl", context_hash="2" * 64)],
                "test": [row(30, context_hash="a" * 64), row(31, context_hash="b" * 64),
                         row(32, question=" ALREADY   SEEN ", context_hash="c" * 64),
                         row(33, question="find item 30", context_hash="d" * 64)]}

    def write_source(self, path):
        manifest = {"complete": True, "dataset": prepare.DATASET, "revision": prepare.REVISION, "files": {}}
        for split, rows in self.sources().items():
            name = prepare.SOURCE_FILES[split]
            pq.write_table(pa.Table.from_pylist(rows), path / name)
            manifest["files"][split] = {"name": name, "rows": len(rows), "sha256": prepare.digest(path / name)}
        prepare.write_json(path / "manifest.json", manifest)

    def test_evaluation_excludes_seen_and_repeated_questions(self):
        sources = self.sources()
        docs = [doc for stage in ("rl", "test") for record in sources[stage]
                for doc in prepare.extract_schema(record)]
        labels, excluded = prepare.make_labels(sources, docs)
        self.assertEqual({split: len(tasks) for split, tasks in labels.items()}, {"train": 1, "dev": 1, "test": 1})
        self.assertEqual({r["reason"] for r in excluded},
                         {"question_seen_in_sft_or_rl", "repeated_evaluation_question"})
        self.assertTrue({r["db_id"] for r in labels["dev"]}.isdisjoint(r["db_id"] for r in labels["test"]))

    def test_complete_bundle_hashes_and_separate_index_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            self.write_source(source)
            output = Path(temporary) / "bundle"
            manifest = prepare.build_bundle(source, output)
            self.assertEqual(manifest["counts"]["tables"], 5)
            self.assertFalse(manifest["live_verification"]["checked"])
            self.assertTrue(all(name.startswith("catalog/") for name in manifest["retrieval_inputs"]))
            for name, info in manifest["files"].items():
                self.assertEqual(prepare.digest(output / name), info["sha256"])
            with self.assertRaises(FileExistsError):
                prepare.build_bundle(source, output)

    def test_corrupt_or_unfinished_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            self.write_source(source)
            with (source / prepare.SOURCE_FILES["test"]).open("ab") as file:
                file.write(b"interrupted download")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare.load_source(source)
            manifest = json.loads((source / "manifest.json").read_text())
            manifest["complete"] = False
            prepare.write_json(source / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "completed"):
                prepare.load_source(source)

    def test_context_overlap_across_source_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            fixtures = self.sources()
            fixtures["test"][0]["context_sha256"] = fixtures["rl"][0]["context_sha256"]
            with patch.object(self, "sources", return_value=fixtures):
                self.write_source(source)
            with self.assertRaisesRegex(ValueError, "context overlap"):
                prepare.load_source(source)


if __name__ == "__main__":
    unittest.main()

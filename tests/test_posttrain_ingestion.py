"""Offline checks for admin SQL confinement and rejection behavior."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pymysql

from data import ingest_mysql_benchmarks as ingest
from infra.verify_mysql_setup import check_eval_grants


CONTEXT = "CREATE TABLE t (id INT, name TEXT); INSERT INTO t VALUES (1, 'a;b');"


class SQLFilterTests(unittest.TestCase):
    def test_original_statements_preserve_quoted_semicolons(self):
        parts, names = ingest.validate_sql(CONTEXT, "SELECT * FROM t;")
        self.assertEqual(parts, ["CREATE TABLE t (id INT, name TEXT)", "INSERT INTO t VALUES (1, 'a;b')"])
        self.assertEqual(names, ["t"])

    def test_cte_and_union_are_readonly_queries(self):
        for query in ("WITH c AS (SELECT * FROM t) SELECT * FROM c", "SELECT id FROM t UNION SELECT id FROM t"):
            ingest.validate_sql(CONTEXT, query)

    def test_admin_context_cannot_escape_row_database(self):
        for context in (
            "DROP DATABASE sandbox_db", "CREATE DATABASE other", "USE sandbox_db",
            "CREATE TABLE sandbox_db.t (id INT)",
            "CREATE TABLE t AS SELECT * FROM sandbox_db.team_check",
            "CREATE TABLE t (id INT); INSERT INTO sandbox_db.team_check VALUES (1)",
            "CREATE TABLE t (id INT); INSERT INTO t SELECT id FROM sandbox_db.team_check",
            "CREATE TABLE t (id INT DEFAULT (GET_LOCK('x', 1)))",
            "CREATE TABLE t (id INT); /*! DROP DATABASE sandbox_db */",
        ):
            with self.subTest(context=context), self.assertRaises(ingest.Rejected):
                ingest.validate_sql(context, "SELECT 1")

    def test_eval_query_cannot_write_or_read_other_databases(self):
        for query in (
            "DELETE FROM t", "SELECT * FROM t; DROP TABLE t", "SELECT * FROM sandbox_db.team_check",
            "SELECT * FROM t FOR UPDATE", "SELECT * INTO OUTFILE '/tmp/result' FROM t",
            "SELECT SLEEP(10)", "SELECT GET_LOCK('x', 10)", "SELECT LOAD_FILE('/etc/passwd')",
        ):
            with self.subTest(query=query), self.assertRaises(ingest.Rejected):
                ingest.validate_sql(CONTEXT, query)

    def test_only_exact_ingestion_database_grants_are_accepted(self):
        for partial in (False, True):
            check_eval_grants([{"grant": 'GRANT SELECT, SHOW VIEW ON "dbpt012345abcdefxtrain000001".* TO "reader"@"%"'}], partial_revokes=partial)
        for database in ("dbpt%", "dbpt012345abcdefxtrain%", "dbpt012345abcdefxtrain000001_other"):
            with self.subTest(database=database), self.assertRaises(RuntimeError):
                check_eval_grants([{"grant": f'GRANT SELECT ON "{database}".* TO "reader"@"%"'}])


class ExecutionTests(unittest.TestCase):
    def worker(self, code=None):
        worker = object.__new__(ingest.Worker)
        worker.prefix = "dbpt012345abcdefx"
        worker.scratch = worker.prefix + "work000000"
        worker.scratch_created = True
        worker.previous_tables = []
        worker.admin = MagicMock()
        worker.eval = MagicMock()
        worker.drop = MagicMock()
        worker.create = MagicMock()
        worker.eval.cursor.return_value.__enter__.return_value.fetchall.return_value = [{"id": 1}]
        if code:
            worker.eval.cursor.return_value.__enter__.return_value.execute.side_effect = pymysql.OperationalError(code, "test")
        return worker

    def row(self):
        return {**dict.fromkeys(ingest.SOURCE_COLUMNS, ""), "id": 1, "sql_context": CONTEXT, "sql": "SELECT * FROM t"}

    def test_eval_credentials_and_readonly_transaction_verify_query(self):
        worker = self.worker()
        record = worker.one("rl", 7, self.row())
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["row"]["db_name"], "dbpt012345abcdefxtrain000007")
        calls = worker.eval.cursor.return_value.__enter__.return_value.execute.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["START TRANSACTION READ ONLY", "SELECT * FROM t"])
        worker.eval.rollback.assert_called_once()
        worker.drop.assert_not_called()

    def test_mysql_row_error_is_rejected_and_live_database_removed(self):
        worker = self.worker(1055)
        record = worker.one("test", 1, self.row())
        self.assertEqual(record["mysql_error_code"], 1055)
        worker.drop.assert_called_once_with("dbpt012345abcdefxtest000001")

    def test_unsupported_stored_function_does_not_gain_execute_privileges(self):
        worker = self.worker(1370)
        record = worker.one("test", 92, self.row())
        self.assertEqual(record["mysql_error_code"], 1370)
        worker.drop.assert_called_once()

    def test_foreign_key_schema_errors_are_rejected(self):
        for code in (1824, 3734, 3780, 6125):
            worker = self.worker(code)
            with self.subTest(code=code):
                record = worker.one("test", 1, self.row())
                self.assertEqual(record["mysql_error_code"], code)

    def test_network_or_storage_failure_stops_instead_of_rejecting_data(self):
        for code in (1044, 1045, 1142, 2006, 2013, 1114):
            worker = self.worker(code)
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, str(code)):
                worker.one("rl", 1, self.row())

    def test_sft_database_is_not_advertised_as_live(self):
        worker = self.worker()
        record = worker.one("sft", 1, self.row())
        self.assertIsNone(record["row"]["db_name"])

    def test_ledger_recovers_only_incomplete_final_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            path.write_bytes(b'{"source_index": 1}\n{"source_')
            self.assertEqual(ingest.read_ledger(path), [{"source_index": 1}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            path.write_text("not-json\n")
            with self.assertRaises(json.JSONDecodeError):
                ingest.read_ledger(path)


class PipelineTests(unittest.TestCase):
    def test_exact_counts_disjoint_contexts_rejections_and_completed_resume(self):
        def row(number, context, query="SELECT 1"):
            return {**dict.fromkeys(ingest.SOURCE_COLUMNS, ""), "id": number,
                    "sql_prompt": f"Question {number}", "sql_context": context, "sql": query}

        source = {
            "test": [row(10, "test context")],
            "train": [row(1, "test context"), row(2, "bad context", "bad"),
                      row(3, "sft context one"), row(4, "sft context two"),
                      row(5, "sft context one"), row(6, "rl context")],
        }

        class FakeWorker:
            calls = 0
            fail_index = 3

            def __init__(self, prefix, *args):
                self.prefix = prefix

            def close(self):
                pass

            def process(self, stage, candidates):
                FakeWorker.calls += 1
                results = []
                split = "test" if stage == "test" else "train"
                for index, item in candidates:
                    record = {"stage": stage, "source_split": split, "source_index": index,
                              "source_id": item["id"]}
                    if split == "train" and index == FakeWorker.fail_index:
                        results.append({**record, "status": "fatal", "reason": "test connection failure"})
                        break
                    if item["sql"] == "bad":
                        results.append({**record, "status": "rejected", "reason": "mysql_execution_error",
                                        "mysql_error_code": 1055})
                    else:
                        clean = {**item, "source_split": split, "source_index": index,
                                 "context_sha256": ingest.fingerprint(item["sql_context"]),
                                 "dataset_revision": ingest.REVISION, "result_row_count": 1,
                                 "db_name": None if stage == "sft" else f"{self.prefix}{split}{index:06d}"}
                        results.append({**record, "status": "accepted", "row": clean})
                return results

        def connection(*args):
            conn = MagicMock()
            responses = iter([
                {"acquired": 1}, {"version": "8.4.11", "sql_mode": "ONLY_FULL_GROUP_BY", "partial_revokes": 1},
            ])
            conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = lambda: next(responses, {"n": 0})
            conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
            conn.__enter__.return_value = conn
            return conn

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(ingest, "Worker", FakeWorker), \
                patch.object(ingest, "connect", side_effect=connection), \
                patch.object(ingest, "required_env", return_value="reader"), \
                patch.object(ingest, "load_dataset", side_effect=lambda *a, split, **kw: source[split]) as loader, \
                patch("builtins.print"):
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "test connection failure"):
                ingest.run(output, "012345abcdef", sft_rows=2, rl_rows=1, test_rows=1,
                           workers=1, checkpoint_rows=2, publish=False)
            saved = json.loads((output / "progress.json").read_text())
            self.assertEqual(saved["counts"], {"sft": 1, "rl": 0, "test": 1})
            self.assertEqual(saved["offsets"], {"train": 3, "test": 1})
            self.assertFalse((output / "ready.json").exists())
            FakeWorker.fail_index = None
            ready = ingest.run(output, "012345abcdef", sft_rows=2, rl_rows=1, test_rows=1,
                               workers=1, checkpoint_rows=2, publish=False)
            self.assertEqual(ready["counts"], {"sft": 2, "rl": 1, "test": 1})
            self.assertFalse(ready["complete"])
            self.assertTrue(ready["ready_to_publish"])
            self.assertFalse(any((output / name).exists() for name in ingest.FILES.values()))
            validated_calls = FakeWorker.calls
            first = ingest.run(output, "012345abcdef", sft_rows=2, rl_rows=1, test_rows=1,
                               workers=1, checkpoint_rows=2)
            self.assertEqual(FakeWorker.calls, validated_calls)
            self.assertTrue(first["complete"])
            self.assertEqual(first["offsets"], {"train": 6, "test": 1})
            self.assertEqual(first["rejections"], {"split_overlap_or_duplicate": 2, "1055": 1})
            tables = {stage: pq.read_table(output / name).to_pylist() for stage, name in ingest.FILES.items()}
            self.assertEqual({stage: len(rows) for stage, rows in tables.items()}, {"sft": 2, "rl": 1, "test": 1})
            self.assertEqual([r["id"] for r in tables["sft"]], [3, 4])
            self.assertEqual(tables["rl"][0]["id"], 6)
            for call in loader.call_args_list:
                self.assertEqual(call.kwargs["revision"], ingest.REVISION)
                self.assertTrue(call.kwargs["streaming"])
            calls_before = FakeWorker.calls
            second = ingest.run(output, "012345abcdef", sft_rows=2, rl_rows=1, test_rows=1,
                                workers=1, checkpoint_rows=2)
            self.assertEqual(FakeWorker.calls, calls_before)
            self.assertEqual(second["files"], first["files"])
            with self.assertRaisesRegex(ValueError, "another run/configuration"):
                ingest.run(output, "aaaaaaaaaaaa", sft_rows=2, rl_rows=1, test_rows=1, workers=1)


if __name__ == "__main__":
    unittest.main()

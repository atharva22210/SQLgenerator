"""Validate original Gretel SQL and publish disjoint post-training Parquet files.

Run locally with ``python -m data.ingest_mysql_benchmarks --help`` or use
``modal run --detach modal_posttrain.py`` for the shared CPU-only job.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pymysql
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

from infra.db_client import connection_options, required_env
from infra.verify_mysql_setup import check_eval_grants


DATASET = "gretelai/synthetic_text_to_sql"
REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"
FILES = {"sft": "sft_train_45k.parquet", "rl": "rl_train_2000.parquet", "test": "eval_test_500.parquet"}
SOURCE_COLUMNS = [
    "id", "domain", "domain_description", "sql_complexity", "sql_complexity_description",
    "sql_task_type", "sql_task_type_description", "sql_prompt", "sql_context", "sql", "sql_explanation",
]
ROW_ERRORS = {1048, 1050, 1052, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1064, 1066, 1067, 1068, 1071,
              1072, 1075, 1109, 1110, 1111, 1118, 1136, 1140, 1146, 1149, 1166, 1170, 1191, 1210,
              1215, 1221, 1222, 1235, 1241, 1242, 1247, 1248, 1250, 1264, 1265, 1273, 1292,
              1305, 1364, 1366, 1370, 1406, 1411, 1452, 1501, 1582, 1630, 1690, 1822,
              1824, 1826, 1830, 1833, 3024, 3065, 3066, 3141, 3592, 3593, 3594,
              3636, 3685, 3734, 3735, 3780, 3814, 3819, 3823, 4025, 6036, 6037, 6125}
# MySQL 8.x recursive-CTE and window-definition semantic errors. These ranges
# deliberately exclude the adjacent temporary-storage configuration errors.
ROW_ERRORS.update(range(3573, 3578))
ROW_ERRORS.update(range(3579, 3598))


class Rejected(ValueError):
    """An unsupported or invalid dataset row, rather than an infra failure."""


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def statements(sql: str) -> list[str]:
    """Split using token offsets, preserving original SQL and quoted semicolons."""
    if not isinstance(sql, str) or not sql.strip():
        raise Rejected("empty_sql")
    # MySQL executable comments can hide statements from a generic AST parser.
    if re.search(r"/\*\s*[!+]", sql):
        raise Rejected("executable_comment_or_hint")
    tokens = sqlglot.Dialect.get_or_raise("mysql").tokenize(sql)
    result, start = [], 0
    for token in tokens:
        if token.token_type == TokenType.SEMICOLON:
            part = sql[start:token.start].strip()
            if part and sqlglot.parse_one(part, read="mysql") is not None:
                result.append(part)
            start = token.end + 1
    tail = sql[start:].strip()
    if tail and sqlglot.parse_one(tail, read="mysql") is not None:
        result.append(tail)
    return result


def validate_sql(context: str, query: str) -> tuple[list[str], list[str]]:
    """Constrain admin execution to local CREATE TABLE and literal INSERTs.

    This parser is a filter; MySQL executes the untouched strings and remains
    the authority on dialect compatibility. Unsupported rows are never fixed.
    """
    try:
        parts = statements(context)
        tables = []
        for part in parts:
            node = sqlglot.parse_one(part, read="mysql")
            for table in node.find_all(exp.Table):
                if table.db or table.catalog or not isinstance(table.this, exp.Identifier):
                    raise Rejected("qualified_context_table")
            if isinstance(node, exp.Create):
                if (node.kind != "TABLE" or not isinstance(node.this, exp.Schema)
                        or node.args.get("expression") or node.args.get("properties")
                        or node.args.get("replace") or node.args.get("clone")):
                    raise Rejected("unsupported_create")
                name = node.this.this.name
                if not name or "`" in name:
                    raise Rejected("unsupported_table_name")
                tables.append(name)
                # Defaults/generated expressions must not invoke arbitrary code.
                if node.find(exp.Func) or node.find(exp.Query):
                    raise Rejected("computed_schema")
            elif isinstance(node, exp.Insert):
                values = node.expression
                if not isinstance(values, exp.Values) or node.find(exp.Query):
                    raise Rejected("nonliteral_insert")
                if node.find(exp.Func) or node.args.get("conflict"):
                    raise Rejected("computed_insert")
                target = node.this.this if isinstance(node.this, exp.Schema) else node.this
                if not isinstance(target, exp.Table) or target.name not in tables:
                    raise Rejected("insert_outside_context")
            else:
                raise Rejected("unsupported_context_statement")
        if not tables:
            raise Rejected("no_tables")
        queries = statements(query)
        if len(queries) != 1:
            raise Rejected("query_must_be_one_select")
        node = sqlglot.parse_one(queries[0], read="mysql")
        if not isinstance(node, exp.Query) or node.find(exp.Into) or node.find(exp.Lock):
            raise Rejected("query_must_be_readonly")
        if node.find(exp.DML) or node.find(exp.DDL) or node.find(exp.Command):
            raise Rejected("query_must_be_readonly")
        for table in node.find_all(exp.Table):
            if table.db or table.catalog:
                raise Rejected("qualified_query_table")
        # SELECT privilege alone still permits these side effects/delays.
        for func in node.find_all(exp.Func):
            name = func.name.upper() if isinstance(func, exp.Anonymous) else func.sql_name()
            if name in {"SLEEP", "BENCHMARK", "GET_LOCK", "RELEASE_LOCK", "LOAD_FILE"}:
                raise Rejected("unsafe_query_function")
        return parts, tables
    except SqlglotError:
        raise Rejected("mysql_parser_rejected") from None


def connect(mode: str):
    options = connection_options(mode)
    options.update(database=None, autocommit=True, read_timeout=30)
    conn = pymysql.connect(**options)
    with conn.cursor() as cur:
        cur.execute("SET SESSION max_execution_time = 5000")
        cur.execute("SET SESSION lock_wait_timeout = 10")
        if mode == "admin":
            # Aiven's documented session override for small synthetic tables;
            # leave the shared service default and original schemas unchanged.
            cur.execute("SET SESSION sql_require_primary_key = 0")
    return conn


def grant_eval(admin, database: str, partial_revokes: bool):
    # Our names have no wildcards, so either MySQL grant interpretation is exact.
    if not re.fullmatch(r"dbpt[0-9a-f]{12}x(?:train|test|work)[0-9]{6}", database):
        raise ValueError("Invalid ingestion database name")
    with admin.cursor() as cur:
        cur.execute(f"GRANT SELECT, SHOW VIEW ON `{database}`.* TO %s@'%%'",
                    (required_env("MYSQL_EVAL_USER"),))


class Worker:
    """Two dedicated connections and one reusable SFT schema per worker."""

    def __init__(self, prefix: str, slot: int, partial_revokes: bool):
        self.admin = connect("admin")
        try:
            self.eval = connect("eval")
        except BaseException:
            self.admin.close()
            raise
        self.prefix = prefix
        self.scratch = f"{prefix}work{slot:06d}"
        self.previous_tables = []
        self.partial_revokes = partial_revokes
        self.scratch_created = False

    def close(self):
        try:
            if self.scratch_created:
                self.drop(self.scratch, revoke=True)
        finally:
            self.eval.close()
            self.admin.close()

    def create(self, name: str):
        with self.admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4")
        grant_eval(self.admin, name, self.partial_revokes)

    def drop(self, name: str, *, revoke: bool = True):
        with self.admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{name}`")
            if revoke:
                cur.execute(f"REVOKE SELECT, SHOW VIEW ON `{name}`.* FROM %s@'%%'",
                            (required_env("MYSQL_EVAL_USER"),))

    def process(self, stage: str, candidates: list[tuple[int, dict]]) -> list[dict]:
        results = []
        for index, row in candidates:
            try:
                results.append(self.one(stage, index, row))
            except (RuntimeError, pymysql.Error) as exc:
                reason = f"MySQL error {exc.args[0]}" if isinstance(exc, pymysql.Error) else str(exc)
                results.append({"stage": stage, "source_split": "test" if stage == "test" else "train",
                                "source_index": index, "source_id": row["id"], "status": "fatal", "reason": reason})
                break
        return results

    def one(self, stage: str, index: int, row: dict) -> dict:
        source = "test" if stage == "test" else "train"
        result = {"stage": stage, "source_split": source, "source_index": index,
                  "source_id": row["id"]}
        try:
            parts, tables = validate_sql(row["sql_context"], row["sql"])
        except Rejected as exc:
            return {**result, "status": "rejected", "reason": str(exc)}
        live = stage != "sft"
        database = f"{self.prefix}{source}{index:06d}" if live else self.scratch
        created = False
        try:
            if live:
                self.create(database)
                created = True
            elif not self.scratch_created:
                self.create(database)
                self.scratch_created = True
            self.admin.select_db(database)
            with self.admin.cursor() as cur:
                if not live and self.previous_tables:
                    cur.execute("SET SESSION foreign_key_checks = 0")
                    try:
                        names = ", ".join(f"`{t}`" for t in sorted(set(self.previous_tables)))
                        cur.execute(f"DROP TABLE IF EXISTS {names}")
                    finally:
                        cur.execute("SET SESSION foreign_key_checks = 1")
                if not live:
                    self.previous_tables = tables
                for part in parts:
                    cur.execute(part)
            self.eval.select_db(database)
            with self.eval.cursor() as cur:
                cur.execute("START TRANSACTION READ ONLY")
                try:
                    cur.execute(row["sql"])
                    # Small synthetic contexts; consume the full result to prove
                    # execution completes, without persisting answer contents.
                    count = len(cur.fetchall())
                finally:
                    self.eval.rollback()
            clean = {key: row[key] for key in SOURCE_COLUMNS}
            clean.update(source_split=source, source_index=index, dataset_revision=REVISION,
                         context_sha256=fingerprint(row["sql_context"]), db_name=database if live else None,
                         result_row_count=count)
            return {**result, "status": "accepted", "row": clean}
        except pymysql.Error as exc:
            code = exc.args[0]
            if created:
                self.drop(database)
            # Access/network/quota/storage errors stop the job, not reject data.
            data_state = (exc.sqlstate or "").startswith(("22", "23"))
            if code not in ROW_ERRORS and not data_state:
                raise RuntimeError(f"Ingestion stopped on MySQL error {code} at {source}:{index}; inspect before resuming") from None
            return {**result, "status": "rejected", "reason": "mysql_execution_error", "mysql_error_code": code}


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_bytes()
    # Only an interrupted final append can be incomplete. A malformed complete
    # line is corruption and must fail rather than silently losing records.
    complete = content.rfind(b"\n") + 1
    if complete != len(content):
        with path.open("r+b") as file:
            file.truncate(complete)
    return [json.loads(line) for line in content[:complete].splitlines()]


def run(output: Path, run_id: str, *, sft_rows=45000, rl_rows=2000, test_rows=500,
        workers=4, checkpoint_rows=1000, publish=True, commit=lambda: None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise ValueError("run-id must be 12 lowercase hexadecimal characters")
    if min(sft_rows, rl_rows, test_rows, workers, checkpoint_rows) <= 0 or workers > 4:
        raise ValueError("Counts must be positive and workers must be at most four")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"dbpt{run_id}x"
    config = {"dataset": DATASET, "revision": REVISION, "run_id": run_id,
              "targets": {"sft": sft_rows, "rl": rl_rows, "test": test_rows}, "workers": workers}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Output belongs to another run/configuration; choose a new output directory")
    elif any(output.iterdir()):
        raise ValueError("New output directory must be empty")
    control = connect("admin")
    started = time.monotonic()
    try:
        with control.cursor() as cur:
            cur.execute("SELECT GET_LOCK(%s, 0) AS acquired", ("sqlgen-posttrain-ingest",))
            if cur.fetchone()["acquired"] != 1:
                raise RuntimeError("Another post-training ingestion holds the MySQL lock")
            cur.execute("SELECT VERSION() AS version, @@sql_mode AS sql_mode, @@partial_revokes AS partial_revokes, "
                        "@@sql_require_primary_key AS session_require_primary_key")
            server = cur.fetchone()
        if not server["version"].startswith(("8.0.", "8.4.")):
            raise ValueError("Expected MySQL 8.0 or the approved MySQL 8.4 server")
        if "ONLY_FULL_GROUP_BY" not in server["sql_mode"].split(","):
            raise ValueError("ONLY_FULL_GROUP_BY must be enabled")
        with connect("eval") as reader, reader.cursor() as cur:
            cur.execute("SHOW GRANTS FOR CURRENT_USER()")
            check_eval_grants(cur.fetchall(), partial_revokes=bool(server["partial_revokes"]))
        if not config_path.exists():
            with control.cursor() as cur:
                cur.execute("SELECT SCHEMA_NAME FROM information_schema.schemata WHERE SCHEMA_NAME LIKE %s",
                            (prefix + "%",))
                if cur.fetchall():
                    raise ValueError("Database namespace already exists; choose a new run-id")
            write_json(config_path, config)
            commit()
        ledger_path = output / "validation.jsonl"
        records = read_ledger(ledger_path)
        accepted = {stage: [] for stage in FILES}
        offsets = {"train": 0, "test": 0}
        contexts = {stage: set() for stage in FILES}
        identities = set()
        reasons = Counter()
        for record in records:
            offsets[record["source_split"]] = max(offsets[record["source_split"]], record["source_index"] + 1)
            if record["status"] == "accepted":
                row, stage = record["row"], record["stage"]
                accepted[stage].append(row)
                contexts[stage].add(row["context_sha256"])
                identities.add(fingerprint(json.dumps([row[k] for k in ("sql_prompt", "sql_context", "sql")])))
            else:
                reasons[str(record.get("mysql_error_code", record["reason"]))] += 1
        # Every pending live database was recorded durably before creation.
        # Only unfinished names in this run are eligible for cleanup on resume.
        pending_path = output / "pending.json"
        completed_names = {r["db_name"] for rows in accepted.values() for r in rows if r["db_name"]}
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())["databases"]
            for database in pending:
                if not re.fullmatch(re.escape(prefix) + r"(?:train|test|work)[0-9]{6}", database):
                    raise ValueError("Invalid pending database namespace")
                if database in completed_names:
                    continue
                with control.cursor() as cur:
                    cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
                    # May have crashed before GRANT; revoke only if it exists.
                    cur.execute("SELECT COUNT(*) AS n FROM information_schema.schema_privileges "
                                "WHERE TABLE_SCHEMA=%s AND GRANTEE=%s",
                                (database, f"'{required_env('MYSQL_EVAL_USER')}'@'%'"))
                    if cur.fetchone()["n"]:
                        cur.execute(f"REVOKE SELECT, SHOW VIEW ON `{database}`.* FROM %s@'%%'",
                                    (required_env("MYSQL_EVAL_USER"),))
        scratch_names = [f"{prefix}work{slot:06d}" for slot in range(workers)]
        write_json(pending_path, {"databases": scratch_names})
        commit()
        streams = {
            source: iter(enumerate(load_dataset(DATASET, revision=REVISION, split=source, streaming=True)))
            for source in ("train", "test")
        }
        for source, stream in streams.items():
            for _ in range(offsets[source]):
                next(stream)
        progress = {"run_id": run_id, "counts": {key: len(rows) for key, rows in accepted.items()},
                    "offsets": offsets, "rejections": dict(reasons), "server": server,
                    "elapsed_seconds": 0, "complete": False}
        with ExitStack() as stack:
            worker_list = []
            for slot in range(workers):
                worker = Worker(prefix, slot, bool(server["partial_revokes"]))
                worker_list.append(worker)
                stack.callback(worker.close)
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
            # Reserve the verified test contexts before selecting any training.
            for stage in ("test", "sft", "rl"):
                source = "test" if stage == "test" else "train"
                target = config["targets"][stage]
                while len(accepted[stage]) < target:
                    control.ping(reconnect=False)  # Preserve ownership of GET_LOCK.
                    candidates, skipped = [], []
                    batch_size = min(checkpoint_rows if stage == "sft" else min(100, checkpoint_rows),
                                     target - len(accepted[stage]))
                    batch_ids = set()
                    forbidden = contexts["test"] | (contexts["sft"] if stage == "rl" else set())
                    while len(candidates) + len(skipped) < batch_size:
                        try:
                            index, row = next(streams[source])
                        except StopIteration:
                            raise RuntimeError(f"Source {source} exhausted before {target} verified {stage} rows") from None
                        identity = fingerprint(json.dumps([row[k] for k in ("sql_prompt", "sql_context", "sql")]))
                        context_hash = fingerprint(row["sql_context"])
                        if identity in identities or identity in batch_ids or (stage != "test" and context_hash in forbidden):
                            skipped.append({"stage": stage, "source_split": source, "source_index": index,
                                            "source_id": row["id"], "status": "rejected", "reason": "split_overlap_or_duplicate"})
                            continue
                        batch_ids.add(identity)
                        candidates.append((index, row))
                    pending_names = scratch_names + ([f"{prefix}{source}{i:06d}" for i, _ in candidates] if stage != "sft" else [])
                    if stage != "sft":
                        write_json(pending_path, {"databases": pending_names})
                        commit()
                    chunks = [candidates[slot::workers] for slot in range(workers)]
                    futures = [pool.submit(worker.process, stage, chunk) for worker, chunk in zip(worker_list, chunks)]
                    results = skipped + [record for future in futures for record in future.result()]
                    results.sort(key=lambda record: record["source_index"])
                    fatal = next((record for record in results if record["status"] == "fatal"), None)
                    if fatal:
                        # Save the contiguous completed prefix. Later successes
                        # stay pending and are cleaned/replayed on resume.
                        results = [record for record in results if record["source_index"] < fatal["source_index"]]
                    with ledger_path.open("a") as ledger:
                        for record in results:
                            if record["status"] == "accepted":
                                row = record["row"]
                                accepted[stage].append(row)
                                contexts[stage].add(row["context_sha256"])
                                identities.add(fingerprint(json.dumps([row[k] for k in ("sql_prompt", "sql_context", "sql")])))
                            else:
                                reasons[str(record.get("mysql_error_code", record["reason"]))] += 1
                            ledger.write(json.dumps(record) + "\n")
                    if results:
                        offsets[source] = max(r["source_index"] for r in results) + 1
                    progress = {"run_id": run_id, "counts": {key: len(rows) for key, rows in accepted.items()},
                                "offsets": offsets, "rejections": dict(reasons), "server": server,
                                "elapsed_seconds": round(time.monotonic() - started, 1), "complete": False}
                    write_json(output / "progress.json", progress)
                    commit()
                    print(json.dumps(progress), flush=True)
                    if fatal:
                        raise RuntimeError(fatal["reason"])
            for left, right in (("sft", "rl"), ("sft", "test"), ("rl", "test")):
                if contexts[left] & contexts[right]:
                    raise AssertionError("Context leakage across output splits")
            manifest = {**config, "server": server, "completed_at": datetime.now(timezone.utc).isoformat(),
                        "rejections": dict(reasons), "offsets": offsets, "files": {},
                        "counts": {key: len(rows) for key, rows in accepted.items()},
                        "ready_to_publish": True, "complete": bool(publish)}
            if publish:
                for stage, rows in accepted.items():
                    path = output / FILES[stage]
                    temporary = path.with_suffix(".parquet.tmp")
                    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
                    temporary.replace(path)
                    actual = pq.read_table(path)
                    if actual.num_rows != config["targets"][stage]:
                        raise AssertionError("Parquet row count mismatch")
                    manifest["files"][stage] = {"name": path.name, "rows": actual.num_rows,
                                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(output / "manifest.json", manifest)
            else:
                write_json(output / "ready.json", manifest)
        write_json(pending_path, {"databases": []})
        write_json(output / "progress.json", {**progress, "ready_to_publish": True, "complete": bool(publish)})
        commit()
        return manifest
    finally:
        control.close()  # Releases the server-side ingestion lock even on error.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="12 lowercase hex characters; reuse to resume")
    parser.add_argument("--sft-rows", type=int, default=45000)
    parser.add_argument("--rl-rows", type=int, default=2000)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_false", dest="publish",
                        help="Save verified checkpoints and counts without writing final Parquet files")
    args = parser.parse_args()
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    try:
        print(json.dumps(run(**vars(args)), indent=2))
    except pymysql.Error as exc:
        raise SystemExit(f"MySQL infrastructure error {exc.args[0]}") from None


if __name__ == "__main__":
    main()

"""Build a schema-only RAG catalogue from the existing verified MySQL corpus.

The catalogue is input to a future retriever. Questions and reference SQL live
in separate label files and must NEVER be embedded into the schema index.
This job collects data; it neither trains a model nor implements retrieval.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from rag.common.catalog import sha256 as digest


SOURCE_FILES = {"sft": "sft_train_45k.parquet", "rl": "rl_train_2000.parquet",
                "test": "eval_test_500.parquet"}
DATASET = "gretelai/synthetic_text_to_sql"
REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"


def normalize(text: str) -> str:
    """Conservative task comparison, not a claim of semantic deduplication."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    """JSONL stores one record per line; catalogue and labels stay separate."""
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source(directory: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Require the published manifest, expected source identity and checksums.

    Args:
        directory: Existing posttrain_data directory, or its local download.
    Returns:
        Original manifest and the three verified lists of source rows.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if (not manifest.get("complete") or manifest.get("dataset") != DATASET
            or manifest.get("revision") != REVISION):
        raise ValueError("Expected the completed, pinned Gretel ingestion")
    sources = {}
    for split, filename in SOURCE_FILES.items():
        info = manifest["files"][split]
        path = directory / filename
        if info["name"] != filename or digest(path) != info["sha256"]:
            raise ValueError(f"Source checksum/filename mismatch: {split}")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != info["rows"] or any(row["dataset_revision"] != REVISION for row in rows):
            raise ValueError(f"Source row count/revision mismatch: {split}")
        sources[split] = rows
    # Identical schemas must not cross source partitions. The preparation job
    # promised this; check it again before relying on that promise for RAG.
    contexts = {split: {r["context_sha256"] for r in rows} for split, rows in sources.items()}
    for a, b in (("sft", "rl"), ("sft", "test"), ("rl", "test")):
        if contexts[a] & contexts[b]:
            raise ValueError(f"Source context overlap: {a}/{b}")
    return manifest, sources


def extract_schema(row: dict) -> list[dict]:
    """Extract CREATE TABLE metadata without executing any source SQL.

    Only DDL supplies relationships. Never infer a foreign key from reference
    SQL: doing so would leak the answer into the retriever's input.
    The returned list contains one structured document per table; INSERT rows
    are deliberately excluded from the searchable text.
    """
    db = row["db_name"]
    if not isinstance(db, str) or not re.fullmatch(r"dbpt[0-9a-f]{12}x(?:train|test)[0-9]{6}", db):
        raise ValueError("Expected a retained ingestion database, not an SFT scratch database")
    documents = []
    for node in sqlglot.parse(row["sql_context"], read="mysql"):
        if not isinstance(node, exp.Create):
            continue
        if node.kind != "TABLE" or not isinstance(node.this, exp.Schema):
            raise ValueError("Unsupported source schema")
        columns, primary, foreign = [], [], []
        for column in node.this.expressions:
            if isinstance(column, exp.ColumnDef):
                columns.append({"name": column.name, "type": column.kind.sql(dialect="mysql")})
                if column.find(exp.PrimaryKeyColumnConstraint):
                    primary.append(column.name)
        for key in node.find_all(exp.PrimaryKey):
            primary.extend(identifier.name for identifier in key.expressions)
        for key in node.find_all(exp.ForeignKey):
            reference = key.args["reference"].this
            foreign.append({"columns": [c.name for c in key.expressions],
                            "referenced_table": reference.this.name,
                            "referenced_columns": [c.name for c in reference.expressions]})
        name = node.this.this.name
        documents.append({
            "document_id": f"{db}.{name}", "db_id": db, "table": name, "dialect": "mysql",
            "domain": row["domain"], "domain_description": row["domain_description"],
            "columns": columns, "primary_key": primary, "foreign_keys": foreign,
            "schema_sql": node.sql(dialect="mysql"), "schema_origin": "source_ddl",
            "source_context_sha256": row["context_sha256"],
        })
    if not documents:
        raise ValueError("Database has no table schemas")
    return documents


def gold_tables(query: str) -> list[str]:
    """Resolve physical tables in nested SQL; CTE/derived aliases are not tables.

    For WITH recent AS (SELECT * FROM orders) SELECT * FROM recent, the gold
    table is orders, not recent. These labels belong only in labels/*.jsonl.
    """
    tree = sqlglot.parse_one(query, read="mysql")
    if not isinstance(tree, exp.Query):
        raise ValueError("Expected a read-only reference query")
    return sorted({source.name for scope in traverse_scope(tree)
                   for source in scope.sources.values() if isinstance(source, exp.Table)})


def snapshot_live_tables(documents: list[dict]) -> dict:
    """Read actual MySQL columns/keys for exactly the selected databases.

    Uses only mysql-eval-secret, SELECT on information_schema and small batches
    of 100 database names. No DDL, INSERT, answer SQL, or admin credentials run.
    Source-vs-live column order and declared constraints must agree. Types from
    MySQL replace source type names (e.g. its normalized representation of REAL).
    """
    import pymysql
    from infra.db_client import connection_options

    expected = {(doc["db_id"], doc["table"].casefold()): doc for doc in documents}
    actual_columns, actual_keys = defaultdict(list), defaultdict(list)
    databases = sorted({doc["db_id"] for doc in documents})
    options = connection_options("eval")
    options.update(database=None, autocommit=True, read_timeout=60)
    with pymysql.connect(**options) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION() AS version")
            server = cursor.fetchone()["version"]
            cursor.execute("SET SESSION max_execution_time = 15000")
            for start in range(0, len(databases), 100):
                names = databases[start:start + 100]
                placeholders = ",".join(["%s"] * len(names))
                cursor.execute(
                    "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, "
                    "COLUMN_COMMENT FROM information_schema.COLUMNS "
                    f"WHERE TABLE_SCHEMA IN ({placeholders}) "
                    "ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION", names)
                for column in cursor.fetchall():
                    actual_columns[(column["TABLE_SCHEMA"], column["TABLE_NAME"].casefold())].append(column)
                cursor.execute(
                    "SELECT TABLE_SCHEMA, TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME, "
                    "REFERENCED_TABLE_SCHEMA, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
                    "FROM information_schema.KEY_COLUMN_USAGE "
                    f"WHERE TABLE_SCHEMA IN ({placeholders}) "
                    "ORDER BY TABLE_SCHEMA, TABLE_NAME, CONSTRAINT_NAME, ORDINAL_POSITION", names)
                for key in cursor.fetchall():
                    actual_keys[(key["TABLE_SCHEMA"], key["TABLE_NAME"].casefold())].append(key)
                print(f"Checked live metadata: {min(start + 100, len(databases))}/{len(databases)} databases", flush=True)
    if set(actual_columns) != set(expected):
        raise ValueError("Live databases/tables differ from the published source catalogue")
    for identity, doc in expected.items():
        columns = actual_columns[identity]
        if [c["name"].casefold() for c in doc["columns"]] != [c["COLUMN_NAME"].casefold() for c in columns]:
            raise ValueError(f"Live columns differ: {doc['document_id']}")
        primary, foreign = [], defaultdict(list)
        for key in actual_keys[identity]:
            if key["CONSTRAINT_NAME"] == "PRIMARY":
                primary.append(key["COLUMN_NAME"])
            if key["REFERENCED_TABLE_NAME"]:
                foreign[key["CONSTRAINT_NAME"]].append(key)
        live_foreign = []
        for parts in foreign.values():
            if any(k["REFERENCED_TABLE_SCHEMA"] != doc["db_id"] for k in parts):
                raise ValueError("Unexpected foreign key outside the selected database")
            live_foreign.append({"columns": [k["COLUMN_NAME"] for k in parts],
                                 "referenced_table": parts[0]["REFERENCED_TABLE_NAME"],
                                 "referenced_columns": [k["REFERENCED_COLUMN_NAME"] for k in parts]})
        canonical = lambda keys: sorted(json.dumps(k, sort_keys=True).casefold() for k in keys)
        if ([c.casefold() for c in primary] != [c.casefold() for c in doc["primary_key"]]
                or canonical(live_foreign) != canonical(doc["foreign_keys"])):
            raise ValueError(f"Live constraints differ: {doc['document_id']}")
        doc.update(columns=[{"name": c["COLUMN_NAME"], "type": c["COLUMN_TYPE"],
                             "nullable": c["IS_NULLABLE"] == "YES", "comment": c["COLUMN_COMMENT"]}
                            for c in columns], primary_key=primary, foreign_keys=live_foreign,
                   schema_origin="live_information_schema")
    return {"checked": True, "server_version": server, "databases": len(databases),
            "tables": len(documents), "checked_at": datetime.now(timezone.utc).isoformat(),
            "method": "read-only information_schema metadata; reference queries not rerun"}


def render_document(doc: dict) -> str:
    """Construct searchable text solely from schema and domain metadata."""
    lines = [f"Database: {doc['db_id']}", f"Domain: {doc['domain']}",
             doc["domain_description"], f"Table: {doc['table']}", "Columns:"]
    lines.extend(f"- {c['name']}: {c['type']}" + (f" ({c['comment']})" if c.get("comment") else "")
                 for c in doc["columns"])
    lines.append("Primary key: " + (", ".join(doc["primary_key"]) or "not declared"))
    lines.append("Declared foreign keys:")
    lines.extend(f"- ({', '.join(k['columns'])}) -> {k['referenced_table']}"
                 f"({', '.join(k['referenced_columns'])})" for k in doc["foreign_keys"])
    if not doc["foreign_keys"]:
        lines.append("- none declared; logical joins may still exist")
    return "\n".join(lines)


def make_labels(sources: dict, documents: list[dict]) -> tuple[dict, list[dict]]:
    """Produce train/dev/test tasks without placing answers in the catalogue.

    RL rows supply future retriever-training labels. Source test rows are split
    by schema hash into development and held-out retrieval test tasks. Exclude
    evaluation questions already appearing in SFT/RL and repeated evaluation
    questions, ignoring case/whitespace. This is stricter than matching answers
    alone, but does not detect paraphrases or prove absence from pretraining.
    """
    by_db = defaultdict(dict)
    for doc in documents:
        by_db[doc["db_id"]][doc["table"].casefold()] = doc["document_id"]
    training_questions = {normalize(row["sql_prompt"]) for split in ("sft", "rl") for row in sources[split]}
    seen_questions = set()
    labels, excluded = {"train": [], "dev": [], "test": []}, []
    for stage in ("rl", "test"):
        for row in sources[stage]:
            question = normalize(row["sql_prompt"])
            reason = None
            if stage == "test" and question in training_questions:
                reason = "question_seen_in_sft_or_rl"
            elif stage == "test" and question in seen_questions:
                reason = "repeated_evaluation_question"
            tables = gold_tables(row["sql"])
            if not tables:
                reason = reason or "query_requires_no_table"
            missing = [table for table in tables if table.casefold() not in by_db[row["db_name"]]]
            if missing:
                raise ValueError(f"Reference query mentions unknown tables: {missing}")
            identity = f"gretel:{row['source_split']}:{row['source_index']}"
            if reason:
                excluded.append({"id": identity, "reason": reason, "source_id": row["id"]})
                continue
            if stage == "test":
                seen_questions.add(question)
            # Split by complete schema, so multiple tasks on one database could
            # never land on opposite sides. This is a RAG project split, NOT the
            # hidden official Gretel test or a new generator-training dataset.
            split = "train" if stage == "rl" else ("dev" if int(row["context_sha256"][:8], 16) % 2 == 0 else "test")
            labels[split].append({
                "id": identity, "question": row["sql_prompt"], "db_id": row["db_name"] ,
                "gold_document_ids": [by_db[row["db_name"]][table.casefold()] for table in tables],
                "gold_tables": tables, "reference_sql": row["sql"], "dialect": "mysql",
                "domain": row["domain"], "source_id": row["id"], "source_split": row["source_split"],
                "source_context_sha256": row["context_sha256"],
                "previous_result_row_count": row["result_row_count"],
            })
    if not all(labels.values()):
        raise ValueError("Preparation must retain nonempty train/dev/test tasks")
    return labels, excluded


def build_bundle(source: Path, output: Path, *, check_live: bool = False) -> dict:
    """Write a new bundle; existing output directories are never overwritten.

    Args:
        source: Verified Parquet files plus their ingestion manifest.
        output: New artifact directory, locally or under /vol/rag on Modal.
        check_live: Read actual schemas with eval credentials before publishing.
    Returns:
        Manifest containing counts, provenance, verification scope and hashes.
    """
    if output.exists():
        raise FileExistsError(f"Choose a new bundle directory: {output}")
    original, sources = load_source(source)
    documents = [doc for stage in ("rl", "test") for row in sources[stage] for doc in extract_schema(row)]
    if len({doc["document_id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate database/table identity")
    live = snapshot_live_tables(documents) if check_live else {"checked": False}
    labels, excluded = make_labels(sources, documents)
    databases, relationships = {}, []
    identities = {doc["document_id"] for doc in documents}
    for doc in documents:
        doc["text"] = render_document(doc)
        database = databases.setdefault(doc["db_id"], {"db_id": doc["db_id"], "domain": doc["domain"],
            "description": doc["domain_description"], "dialect": "mysql", "table_document_ids": []})
        database["table_document_ids"].append(doc["document_id"])
        for key in doc["foreign_keys"]:
            target = f"{doc['db_id']}.{key['referenced_table']}"
            if target not in identities:
                raise ValueError(f"Foreign key references an unknown table: {target}")
            relationships.append({"from_document_id": doc["document_id"], "to_document_id": target,
                                  **key, "evidence": "declared_foreign_key"})
    output.mkdir(parents=True)
    (output / "catalog").mkdir()
    (output / "labels").mkdir()
    write_rows(output / "catalog/tables.jsonl", documents)
    write_rows(output / "catalog/databases.jsonl", list(databases.values()))
    write_rows(output / "catalog/relationships.jsonl", relationships)
    for split, rows in labels.items():
        write_rows(output / f"labels/{split}.jsonl", rows)
    write_rows(output / "labels/excluded.jsonl", excluded)
    # A readable trace shows exactly what the future retriever may see and
    # what only the evaluator may see. It is outside catalog/ on purpose.
    samples = []
    lookup = {doc["document_id"]: doc for doc in documents}
    for split in ("train", "dev", "test"):
        selected = labels[split][:1] + [row for row in labels[split] if len(row["gold_tables"]) > 1][:1]
        for row in selected:
            samples.extend([f"## {split}: {row['id']}", f"Question: {row['question']}",
                            f"Database: {row['db_id']}", "\nRetrieved-schema target (oracle):"])
            samples.extend("```text\n" + lookup[identity]["text"] + "\n```" for identity in row["gold_document_ids"])
            samples.append("Evaluator-only reference SQL:\n```sql\n" + row["reference_sql"] + "\n```\n")
    (output / "samples.md").write_text("# RAG data examples — do not index this file\n\n" + "\n\n".join(samples))
    write_json(output / "source_manifest.json", original)
    manifest = {
        "format_version": 1, "bundle_id": output.name, "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}", "license": "Apache-2.0",
        "revision": REVISION, "source_volume": "post-train-data", "source_directory": "posttrain_data",
        "source_manifest_sha256": digest(source / "manifest.json"), "live_verification": live,
        "counts": {"databases": len(databases), "tables": len(documents), "declared_foreign_keys": len(relationships),
                   "multi_table_databases": sum(len(db["table_document_ids"]) > 1 for db in databases.values()),
                   **{f"{split}_questions": len(rows) for split, rows in labels.items()},
                   "excluded_evaluation_or_tableless_tasks": len(excluded)},
        "excluded_reasons": dict(Counter(row["reason"] for row in excluded)),
        "retrieval_inputs": ["catalog/tables.jsonl", "catalog/databases.jsonl", "catalog/relationships.jsonl"],
        "evaluation_policy": {
            "split": "RL -> train; source test -> schema-hash dev/test",
            "overlap_check": "normalized question absent from 45k SFT and 2k RL; unique evaluation question",
            "pretraining_overlap_checked": False, "semantic_overlap_checked": False,
            "db_scoped_mode": "Evaluator supplies db_id; retriever sees question plus that database's catalogue",
            "global_mode": "Retriever sees question only; repeated domains can make database routing ambiguous",
        },
        "limitations": ["Synthetic independent databases, often one table; not an enterprise warehouse",
                        "Only explicitly declared foreign keys are indexed; absent edges do not mean no logical joins",
                        "Reference SQL was executed during source ingestion, not certified semantically correct",
                        "Does not prove generator test independence from prior tokenized pretraining"],
        "files": {},
    }
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    # Written last: consumers must reject bundles without this completion file.
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/rag/source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.source, args.output, check_live=args.check_live), indent=2))


if __name__ == "__main__":
    main()

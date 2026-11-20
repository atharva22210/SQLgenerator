"""Retrieve tables with plain hybrid, then reduce each table to scored columns."""
import json
from pathlib import Path
import time

import numpy as np

from rag.common.prepare_data import render_document
from rag.common.storage import save_json
from rag.dense.encoder import Encoder
from rag.dense.index import normalize_rows
from rag.hybrid import HybridIndex
from rag.hybrid_columns.selection import column_text, render_schema_sql, required_columns

VERSION = 1
COLUMN_TEXT_VERSION = "table_column_domain_v1"


class HybridColumnsIndex:
    """Use hybrid's top-k tables, then select top columns per table plus keys.

    hybrid: Verified plain hybrid index; graph expansion is not used.
    columns_per_table: Semantic selections per table (1..100), BEFORE adding
        mandatory PK/FKs. This is not a hard cap on the final column count.
    batch_size: Build-time column encoding batch size. column_embeddings: Saved
        [total_columns,H] matrix in table/column order, reused when loading.
    """
    def __init__(self, hybrid: HybridIndex, *, columns_per_table: int = 3, batch_size: int = 32,
                 column_embeddings=None):
        if (isinstance(columns_per_table, bool) or not isinstance(columns_per_table, int)
                or not 1 <= columns_per_table <= 100):
            raise ValueError("columns_per_table must be an integer in [1,100]")
        self.hybrid, self.columns_per_table = hybrid, columns_per_table
        self.documents, self.by_db, self.source_sha256 = hybrid.documents, hybrid.by_db, hybrid.source_sha256
        self.lookup = {d["document_id"]: d for d in self.documents}
        self.column_records, self.positions, texts = [], {}, []
        for doc in self.documents:
            columns = doc["columns"]
            if not columns or len({c["name"] for c in columns}) != len(columns):
                raise ValueError("Require nonempty columns with unique names per table")
            start = len(texts)
            for column in columns:
                self.column_records.append([doc["document_id"], column["name"]])
                texts.append(column_text(doc, column))
            self.positions[doc["document_id"]] = list(range(start, len(texts)))
        if column_embeddings is None:
            column_embeddings = hybrid.dense.encoder.encode(texts, batch_size=batch_size)
        self.embeddings = normalize_rows(column_embeddings, rows=len(texts))
        if self.embeddings.shape[1] != hybrid.dense.embeddings.shape[1]:
            raise ValueError("Column and table embedding dimensions differ")

    def configuration(self) -> dict:
        """Record semantic selection size, text recipe and shared hybrid settings."""
        return {"method": "hybrid_columns", "columns_per_table": self.columns_per_table,
                "column_text_version": COLUMN_TEXT_VERSION, "hybrid": self.hybrid.configuration()}

    def search(self, query: str, *, k: int = 4, db_id: str | None = None,
               allowed_db_ids: set[str] | None = None, explain: bool = False) -> dict:
        """Return k ranked tables with reduced schema_text AND schema_sql.

        Hybrid applies permission/scope validation before selection. Column
        scoring only visits those returned tables; schema order is preserved
        after selection. Full original schemas are not copied into the response.
        """
        started = time.perf_counter()
        response = self.hybrid.search(query, k=k, db_id=db_id, allowed_db_ids=allowed_db_ids, explain=explain)
        selected_docs = [self.lookup[hit["document_id"]] for hit in response["results"]]
        required = required_columns(selected_docs)
        vector = None
        if selected_docs:
            vector = normalize_rows(self.hybrid.dense.encoder.encode([query]), rows=1)
            if vector.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Query and column embedding dimensions differ")
        results, before, after = [], 0, 0
        for hit, doc in zip(response["results"], selected_docs):
            scores = self.embeddings[self.positions[doc["document_id"]]] @ vector[0]
            ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), doc["columns"][i]["name"]))
            semantic = {doc["columns"][i]["name"] for i in ranked[:self.columns_per_table]}
            mandatory = required[doc["document_id"]]
            columns = [c for c in doc["columns"] if c["name"] in semantic | mandatory]
            reduced = {**doc, "columns": columns}
            selected = {**hit, "columns": columns, "primary_key": doc["primary_key"],
                        "foreign_keys": doc["foreign_keys"], "schema_text": render_document(reduced),
                        "schema_sql": render_schema_sql(reduced)}
            selected["column_selection"] = {"original_count": len(doc["columns"]), "selected_count": len(columns),
                                             "mandatory_columns": sorted(mandatory)}
            if explain:
                selected["column_selection"]["scores"] = [{"column": c["name"], "score": float(scores[i]),
                    "semantic_selection": c["name"] in semantic, "required_key": c["name"] in mandatory,
                    "retained": c["name"] in semantic | mandatory} for i, c in enumerate(doc["columns"])]
            results.append(selected)
            before, after = before + len(doc["columns"]), after + len(columns)
        return {**response, "method": "hybrid_columns", "results": results,
                "columns_before": before, "columns_after": after,
                "elapsed_ms": (time.perf_counter() - started) * 1000}

    def save(self, path: Path) -> None:
        """Save shared hybrid plus column-vector order and recipe in one new file."""
        save_json(path, {"version": VERSION, **self.configuration(), "hybrid": self.hybrid.snapshot(),
                         "column_records": self.column_records, "column_embeddings": self.embeddings.tolist()})

    @classmethod
    def load(cls, path: Path, bundle: Path, *, encoder: Encoder | None = None, device: str = "cpu"):
        """Verify source and column order, reuse saved embeddings without encoding."""
        payload = json.loads(path.read_text())
        if (payload["version"] != VERSION or payload["method"] != "hybrid_columns"
                or payload["column_text_version"] != COLUMN_TEXT_VERSION):
            raise ValueError("Stale/incompatible column index; rebuild into a new file")
        hybrid = HybridIndex.from_snapshot(payload["hybrid"], bundle, encoder=encoder, device=device)
        expected = [[d["document_id"], c["name"]] for d in hybrid.documents for c in d["columns"]]
        if payload["column_records"] != expected:
            raise ValueError("Saved column order disagrees with catalogue")
        return cls(hybrid, columns_per_table=payload["columns_per_table"],
                   column_embeddings=payload["column_embeddings"])

"""Offline checks for column retention and joint question-table reranking."""
from contextlib import redirect_stdout
import http.client
from http.server import HTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import sqlglot
import torch

from rag.bm25 import BM25Index
from rag.common.catalog import sha256
from rag.common.cli import main
from rag.common.evaluation import evaluate
from rag.common.prepare_data import render_document
from rag.common.service import make_handler
from rag.dense import DenseIndex
from rag.hybrid import HybridIndex
from rag.hybrid_columns import HybridColumnsIndex
from rag.hybrid_columns.selection import render_schema_sql
from rag.hybrid_rerank import HybridRerankIndex
from rag.hybrid_rerank.model import MiniLMCrossEncoder


def documents(db="shop"):
    docs = [{"table": "customers", "columns": ["id", "tenant", "code", "name", "email", "unused"],
             "primary_key": ["id", "tenant"], "foreign_keys": []},
            {"table": "orders", "columns": ["id", "customer_code", "customer_tenant", "total", "internal_note", "legacy"],
             "primary_key": ["id"], "foreign_keys": [{"columns": ["customer_code", "customer_tenant"],
                 "referenced_table": "customers", "referenced_columns": ["code", "tenant"]}]}]
    for doc in docs:
        doc.update(document_id=f"{db}.{doc['table']}", db_id=db, domain="retail", domain_description="", dialect="mysql")
        doc["columns"] = [{"name": name, "type": "INT"} for name in doc["columns"]]
        doc["text"], doc["schema_sql"] = render_document(doc), render_schema_sql(doc)
    return docs


class FixtureEncoder:
    def __init__(self):
        self.seen = []

    def configuration(self):
        return {"model_id": "refinements-fixture", "revision": "1"}

    def encode(self, texts, *, batch_size=32):
        self.seen.extend(texts)
        # Only name/total score highly; all join keys deliberately score poorly.
        return np.array([[1, 0] if ". Column " not in text or
                         text.split(". Column ")[1].split()[0] in {"name", "total"} else [-1, 0]
                         for text in texts], dtype=np.float32)


class FixtureReranker:
    def __init__(self):
        self.seen = []

    def configuration(self):
        return {"model_id": "pair-fixture", "revision": "1"}

    def score(self, query, passages):
        self.seen.extend((query, passage) for passage in passages)
        return np.array([2 if passage.startswith("orders ") else -1 for passage in passages], dtype=np.float32)


def write_bundle(root, docs):
    (root / "catalog").mkdir()
    path = root / "catalog/tables.jsonl"
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    checksum = sha256(path)
    (root / "manifest.json").write_text(json.dumps({"complete": True, "counts": {"tables": len(docs)},
        "files": {"catalog/tables.jsonl": {"sha256": checksum}}}))
    return checksum


class ColumnTests(unittest.TestCase):
    def setUp(self):
        self.docs, self.encoder = documents(), FixtureEncoder()
        self.hybrid = HybridIndex(BM25Index(self.docs), DenseIndex(self.docs, self.encoder))
        self.index = HybridColumnsIndex(self.hybrid, columns_per_table=1)

    def test_low_scoring_composite_keys_and_non_pk_reference_are_kept(self):
        result = self.index.search("buyers", k=2, explain=True)
        hits = {h["table"]: h for h in result["results"]}
        self.assertEqual({c["name"] for c in hits["customers"]["columns"]}, {"id", "tenant", "code", "name"})
        self.assertEqual({c["name"] for c in hits["orders"]["columns"]}, {"id", "customer_code", "customer_tenant", "total"})
        self.assertEqual(result["columns_before"], 12)
        self.assertEqual(result["columns_after"], 8)
        for hit in hits.values():
            parsed = sqlglot.parse_one(hit["schema_sql"], read="mysql")
            self.assertEqual([c.name for c in parsed.this.expressions if isinstance(c, sqlglot.exp.ColumnDef)],
                             [c["name"] for c in hit["columns"]])
        self.assertNotIn("internal_note", hits["orders"]["schema_text"])
        self.assertNotIn("internal_note", hits["orders"]["schema_sql"])
        self.assertEqual(len(self.docs[1]["columns"]), 6)  # Never mutate shared catalogue.
        self.assertLess(sum(len(h["schema_text"]) for h in hits.values()), sum(len(d["text"]) for d in self.docs))

    def test_scope_and_empty_results_do_not_score_private_columns(self):
        docs = self.docs + documents("secret")
        encoder = FixtureEncoder()
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, encoder))
        index = HybridColumnsIndex(hybrid, columns_per_table=1)
        result = index.search("buyers", k=2, allowed_db_ids={"shop"})
        self.assertEqual({hit["db_id"] for hit in result["results"]}, {"shop"})
        encoder.seen.clear()
        self.assertEqual(index.search("buyers", allowed_db_ids=set())["results"], [])
        self.assertEqual(encoder.seen, [])
        with self.assertRaises(ValueError):
            index.search("buyers", db_id="secret", allowed_db_ids={"shop"})

    def test_snapshot_reuses_columns_and_rejects_wrong_order_and_stale_catalogue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checksum = write_bundle(root, self.docs)
            self.hybrid.bm25.source_sha256 = self.hybrid.dense.source_sha256 = self.hybrid.source_sha256 = checksum
            path = root / "columns.json"
            self.index.save(path)
            encoder = FixtureEncoder()
            loaded = HybridColumnsIndex.load(path, root, encoder=encoder)
            self.assertEqual(encoder.seen, [])
            self.assertEqual(loaded.search("buyers", k=2)["results"], self.index.search("buyers", k=2)["results"])
            with self.assertRaises(FileExistsError):
                self.index.save(path)
            payload = json.loads(path.read_text())
            payload["column_records"].reverse()
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                HybridColumnsIndex.load(path, root, encoder=encoder)
            (root / "catalog/tables.jsonl").write_text("corrupt")
            with self.assertRaises(ValueError):
                HybridColumnsIndex.load(path, root, encoder=encoder)

    def test_column_bounds_and_dimension_validation(self):
        for count in (0, True, 101):
            with self.assertRaises(ValueError):
                HybridColumnsIndex(self.hybrid, columns_per_table=count)
        with self.assertRaises(ValueError):
            HybridColumnsIndex(self.hybrid, column_embeddings=np.ones((12, 3)))


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.docs, self.encoder, self.reranker = documents(), FixtureEncoder(), FixtureReranker()
        self.hybrid = HybridIndex(BM25Index(self.docs), DenseIndex(self.docs, self.encoder))
        self.index = HybridRerankIndex(self.hybrid, self.reranker, candidate_pool=2)

    def test_joint_reranking_reverses_order_and_evaluation_scores_final_order(self):
        self.assertEqual(self.hybrid.search("buyers", k=1)["results"][0]["table"], "customers")
        response = self.index.search("buyers", k=1, explain=True)
        self.assertEqual(response["results"][0]["table"], "orders")
        self.assertEqual(response["results"][0]["score"], 2)
        self.assertEqual(response["results"][0]["reranking"]["hybrid_rank"], 2)
        with tempfile.TemporaryDirectory() as temp:
            labels = Path(temp) / "dev.jsonl"
            labels.write_text(json.dumps({"id": "q", "question": "buyers", "db_id": "shop",
                                          "gold_document_ids": ["shop.orders"]}) + "\n")
            self.reranker.seen.clear()
            report = evaluate(self.index, labels, k=1, scoped=True)
            self.assertEqual(report["all"]["recall"], 1)
            self.assertEqual([q for q, _ in self.reranker.seen], ["buyers", "buyers"])
            self.assertTrue(all("shop.orders" not in text for _, text in self.reranker.seen))

    def test_permissions_pool_cap_ties_and_nonfinite_scores(self):
        for size in (0, True, 21):
            with self.assertRaises(ValueError):
                HybridRerankIndex(self.hybrid, self.reranker, candidate_pool=size)
        docs = self.docs + documents("secret")
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, self.encoder))
        index = HybridRerankIndex(hybrid, self.reranker, candidate_pool=2)
        self.assertEqual({hit["db_id"] for hit in index.search("buyers", k=2, allowed_db_ids={"shop"})["results"]}, {"shop"})
        self.reranker.seen.clear()
        self.assertEqual(index.search("buyers", k=2, allowed_db_ids=set())["results"], [])
        self.assertEqual(self.reranker.seen, [])
        with self.assertRaises(ValueError):
            index.search("buyers", k=3)
        with patch.object(self.reranker, "score", return_value=np.ones(2)):
            self.assertEqual(index.search("buyers", k=1, db_id="shop")["results"][0]["table"], "customers")
        for scores in (np.array([float("nan"), 1]), np.ones(1), np.ones((2, 1))):
            with patch.object(self.reranker, "score", return_value=scores):
                with self.assertRaises(ValueError):
                    index.search("buyers", k=1, db_id="shop")

    def test_snapshot_rejects_changed_model_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checksum = write_bundle(root, self.docs)
            self.hybrid.bm25.source_sha256 = self.hybrid.dense.source_sha256 = self.hybrid.source_sha256 = checksum
            path = root / "rerank.json"
            self.index.save(path)
            encoder = FixtureEncoder()
            loaded = HybridRerankIndex.load(path, root, encoder=encoder, reranker=FixtureReranker())
            self.assertEqual(encoder.seen, [])
            self.assertEqual(loaded.search("buyers", k=1)["results"], self.index.search("buyers", k=1)["results"])
            payload = json.loads(path.read_text())
            payload["reranker"]["revision"] = "different"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                HybridRerankIndex.load(path, root, encoder=encoder, reranker=FixtureReranker())


class IntegrationTests(unittest.TestCase):
    def test_cli_build_search_evaluate_for_both_backends_without_downloads(self):
        for backend in ("hybrid_columns", "hybrid_rerank"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_bundle(root, documents())
                (root / "labels").mkdir()
                (root / "labels/dev.jsonl").write_text(json.dumps({"id": "q", "question": "buyers", "db_id": "shop",
                    "gold_document_ids": ["shop.orders"]}) + "\n")
                def run(tail):
                    output = io.StringIO()
                    prefix = ["rag", "--bundle", str(root), "--index", str(root / "index.json")]
                    with patch("sys.argv", prefix + tail), redirect_stdout(output):
                        main(default_backend=backend)
                    return json.loads(output.getvalue())
                with patch("rag.dense.encoder.MiniLMEncoder", return_value=FixtureEncoder()), \
                     patch("rag.dense.index.MiniLMEncoder", return_value=FixtureEncoder()), \
                     patch("rag.dense.index.encoder_configuration", return_value=FixtureEncoder().configuration()), \
                     patch("rag.hybrid_rerank.model.MiniLMCrossEncoder", return_value=FixtureReranker()), \
                     patch("rag.hybrid_rerank.index.MiniLMCrossEncoder", return_value=FixtureReranker()), \
                     patch("rag.hybrid_rerank.index.reranker_configuration", return_value=FixtureReranker().configuration()):
                    self.assertEqual(run(["build"])["tables"], 2)
                    self.assertEqual(run(["search", "--query", "buyers"])["method"], backend)
                    report = run(["evaluate", "--output", str(root / "report.json")])
                    self.assertEqual(report["method"], backend)
                    self.assertEqual(report["k"], 4)

    def test_shared_http_handler_serves_both_refinements(self):
        docs, encoder = documents(), FixtureEncoder()
        hybrid = HybridIndex(BM25Index(docs), DenseIndex(docs, encoder))
        for index in (HybridColumnsIndex(hybrid, columns_per_table=1),
                      HybridRerankIndex(hybrid, FixtureReranker(), candidate_pool=8)):
            server = HTTPServer(("127.0.0.1", 0), make_handler(index, "t" * 32, {"shop"}))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                connection.request("POST", "/search", json.dumps({"query": "buyers"}),
                    {"Content-Type": "application/json", "Authorization": "Bearer " + "t" * 32})
                response = connection.getresponse()
                payload = json.loads(response.read())
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["method"], index.configuration()["method"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_real_cross_encoder_pair_forward_is_batched_and_frozen(self):
        from transformers import BertConfig, BertForSequenceClassification

        model = BertForSequenceClassification(BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=1))
        seen = []
        def tokenizer(questions, passages, **kwargs):
            seen.extend(zip(questions, passages))
            self.assertEqual(kwargs["max_length"], 512)
            return {"input_ids": torch.tensor([[1, 2, 3]] * len(passages)),
                    "attention_mask": torch.ones((len(passages), 3), dtype=torch.long),
                    "token_type_ids": torch.tensor([[0, 0, 1]] * len(passages))}
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model):
            reranker = MiniLMCrossEncoder(batch_size=2)
            scores = reranker.score("question", ["table1", "table2", "table3"])
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(seen, [("question", "table1"), ("question", "table2"), ("question", "table3")])
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

"""Offline hybrid rank mathematics, permissions, snapshots and CLI integration."""
from contextlib import redirect_stdout
import io
import http.client
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from rag.bm25 import BM25Index
from rag.common.catalog import sha256
from rag.common.cli import main
from rag.common.evaluation import evaluate
from rag.common.service import make_handler
from rag.dense import DenseIndex
from rag.hybrid import HybridIndex


def document(identity, words, db="shop"):
    return {"document_id": identity, "db_id": db, "table": words, "domain": "",
            "domain_description": "", "columns": [], "primary_key": [], "foreign_keys": [],
            "text": words, "schema_sql": "CREATE TABLE example (id INT)"}


class FixtureEncoder:
    """Vectors produce deliberately different rankings from word frequencies."""
    def __init__(self):
        self.seen = []

    def configuration(self):
        return {"model_id": "hybrid-offline-fixture", "revision": "1"}

    def encode(self, texts, *, batch_size=32):
        self.seen.extend(texts)
        vectors = {"apple apple": [0, 1], "apple": [1, 0], "pear": [.8, .6],
                   "apple apple apple": [1, 0], "buyers": [1, 0]}
        return np.array([vectors[text.strip()] for text in texts], dtype=np.float32)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.docs = [document("a", "apple apple"), document("b", "apple"), document("c", "pear"),
                     document("private", "apple apple apple", "secret")]
        self.encoder = FixtureEncoder()
        self.index = HybridIndex(BM25Index(self.docs), DenseIndex(self.docs, self.encoder))

    def bundle(self, root):
        (root / "catalog").mkdir()
        tables = root / "catalog/tables.jsonl"
        tables.write_text("".join(json.dumps(doc) + "\n" for doc in self.docs))
        checksum = sha256(tables)
        (root / "manifest.json").write_text(json.dumps({"complete": True, "counts": {"tables": len(self.docs)},
            "files": {"catalog/tables.jsonl": {"sha256": checksum}}}))
        return checksum

    def test_rrf_hand_calculation_and_pool_larger_than_final_k(self):
        response = self.index.search("apple", k=1, allowed_db_ids={"shop"}, explain=True)
        hit = response["results"][0]
        # BM25 ranks a,b; dense ranks b,c,a. Agreement makes b the winner,
        # even though it is not BM25's first hit and final k is only one.
        self.assertEqual(hit["document_id"], "b")
        self.assertAlmostEqual(hit["score"], 1/62 + 1/61)
        children = hit["explanation"]["children"]
        self.assertEqual(children["bm25"]["rank"], 2)
        self.assertEqual(children["dense"]["rank"], 1)
        self.assertAlmostEqual(sum(child["contribution"] for child in children.values()), hit["score"])
        self.assertEqual(response["candidate_count"], 3)
        self.assertEqual(response["matched_documents"], 3)

    def test_permissions_empty_scope_and_dense_only_matches(self):
        response = self.index.search("buyers", allowed_db_ids={"shop"}, explain=True)
        self.assertEqual(response["results"][0]["document_id"], "b")
        self.assertEqual(set(response["results"][0]["explanation"]["children"]), {"dense"})
        self.assertAlmostEqual(response["results"][0]["score"], 1/61)
        self.encoder.seen.clear()
        self.assertEqual(self.index.search("apple", allowed_db_ids=set())["results"], [])
        self.assertEqual(self.encoder.seen, [])
        with self.assertRaises(ValueError):
            self.index.search("apple", db_id="secret", allowed_db_ids={"shop"})
        for query, k in (("", 1), ("apple", True), ("apple", 21)):
            with self.assertRaises(ValueError):
                self.index.search(query, k=k)

    def test_rank_fusion_ignores_raw_score_scale_and_breaks_ties_by_id(self):
        bm25_hits = [{"document_id": "b", "score": 1000000}, {"document_id": "a", "score": 100}]
        dense_hits = [{"document_id": "a", "score": .9}, {"document_id": "b", "score": .1}]
        with patch.object(self.index.bm25, "search", return_value={"results": bm25_hits}), \
             patch.object(self.index.dense, "search", return_value={"results": dense_hits}):
            hits = self.index.search("apple")["results"]
        self.assertEqual([hit["document_id"] for hit in hits], ["a", "b"])
        self.assertEqual(hits[0]["score"], hits[1]["score"])

    def test_requires_matching_children_and_valid_constant(self):
        for constant in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                HybridIndex(self.index.bm25, self.index.dense, rrf_constant=constant)
        with self.assertRaises(ValueError):
            HybridIndex(BM25Index(self.docs, "different"), self.index.dense)
        with self.assertRaises(ValueError):
            HybridIndex(BM25Index(list(reversed(self.docs))), self.index.dense)

    def test_snapshot_reuses_embeddings_and_rejects_stale_recipes_and_catalogue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checksum = self.bundle(root)
            self.index.bm25.source_sha256 = self.index.dense.source_sha256 = self.index.source_sha256 = checksum
            path = root / "index.json"
            self.index.save(path)
            fresh = FixtureEncoder()
            loaded = HybridIndex.load(path, root, encoder=fresh)
            self.assertEqual(fresh.seen, [])
            self.assertEqual(loaded.search("apple")["results"], self.index.search("apple")["results"])
            with self.assertRaises(FileExistsError):
                self.index.save(path)
            original = path.read_text()
            for child, field in (("bm25", "analyzer"), ("dense", "representation")):
                payload = json.loads(original)
                payload[child][field] = "different"
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    HybridIndex.load(path, root, encoder=fresh)
            path.write_text(original)
            (root / "catalog/tables.jsonl").write_text("corrupt")
            with self.assertRaises(ValueError):
                HybridIndex.load(path, root, encoder=fresh)

    def test_cli_and_evaluation_reuse_common_paths_without_gold_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.bundle(root)
            (root / "labels").mkdir()
            labels = root / "labels/dev.jsonl"
            labels.write_text(json.dumps({"id": "q", "question": "apple", "db_id": "shop",
                                          "gold_document_ids": ["b"]}) + "\n")
            prefix = ["rag", "--bundle", str(root), "--index", str(root / "index.json")]
            def run(tail):
                output = io.StringIO()
                with patch("sys.argv", prefix + tail), redirect_stdout(output):
                    main(default_backend="hybrid")
                return json.loads(output.getvalue())
            with patch("rag.dense.encoder.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.encoder_configuration", return_value=FixtureEncoder().configuration()):
                self.assertEqual(run(["build"])["tables"], 4)
                result = run(["search", "--query", "apple", "--db-id", "shop", "--k", "1"])
                self.assertEqual(result["results"][0]["document_id"], "b")
                report = root / "report.json"
                self.assertEqual(run(["evaluate", "--scoped", "--k", "1", "--output", str(report)])
                                 ["all"]["recall"], 1)
            self.encoder.seen.clear()
            evaluation = evaluate(self.index, labels, k=1, scoped=True)
            self.assertEqual(self.encoder.seen, ["apple"])
            self.assertEqual(evaluation["method"], "hybrid")
            self.assertEqual(evaluation["all"]["recall"], 1)

    def test_shared_http_api_applies_allowlist_to_both_children(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.index, "t" * 32, {"shop"}))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            connection.request("POST", "/search", json.dumps({"query": "apple", "k": 1, "explain": True}),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + "t" * 32})
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["candidate_count"], 3)
            self.assertEqual(payload["results"][0]["document_id"], "b")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

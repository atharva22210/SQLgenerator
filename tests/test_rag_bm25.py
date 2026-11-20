"""BM25 mathematics, permission filtering, snapshot integrity and API behavior."""
import http.client
from http.server import HTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

from rag.bm25 import BM25Index, tokenize, sha256
from rag.cli import evaluate
from rag.service import make_handler


def document(identity, words, db="shop"):
    return {"document_id": identity, "db_id": db, "table": words, "domain": "",
            "domain_description": "", "columns": [], "primary_key": [], "foreign_keys": [],
            "text": words, "schema_sql": "CREATE TABLE example (id INT)"}


class RetrievalTests(unittest.TestCase):
    def test_identifier_analysis(self):
        self.assertEqual(tokenize("customer_id customerId CAFÉ"), ["customer", "id", "customer", "id", "café"])

    def test_bm25_matches_hand_calculation(self):
        index = BM25Index([document("a", "apple apple"), document("b", "pear")])
        response = index.search("apple", explain=True)
        expected = math.log(2) * (2 * 2.2) / (2 + 1.2 * (.25 + .75 * 2 / 1.5))
        self.assertAlmostEqual(response["results"][0]["score"], expected)
        self.assertAlmostEqual(sum(response["results"][0]["term_scores"].values()), expected)
        self.assertEqual(len(response["results"]), 1)
        self.assertEqual(index.search("unknown")["status"], "no_lexical_match")

    def test_permission_filter_precedes_top_k_and_ties_are_stable(self):
        index = BM25Index([document("z", "apple", "private"), document("b", "apple"), document("a", "apple")])
        result = index.search("apple", k=1, allowed_db_ids={"shop"})
        self.assertEqual(result["results"][0]["document_id"], "a")
        with self.assertRaises(ValueError):
            index.search("apple", db_id="private", allowed_db_ids={"shop"})
        self.assertEqual(index.search("apple", allowed_db_ids=set())["results"], [])

    def test_request_and_parameter_bounds(self):
        for k1, b in ((0, .75), (float("nan"), .75), (1.2, 2)):
            with self.assertRaises(ValueError):
                BM25Index([document("a", "apple")], k1=k1, b=b)
        index = BM25Index([document("a", "apple")])
        for query, k in (("", 1), ("a" * 2001, 1), ("apple", True), ("apple", 21)):
            with self.assertRaises(ValueError):
                index.search(query, k=k)

    def test_snapshot_roundtrip_no_clobber_and_stale_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "catalog").mkdir()
            table = root / "catalog/tables.jsonl"
            docs = [document("a", "apple")]
            table.write_text(json.dumps(docs[0]) + "\n")
            (root / "manifest.json").write_text(json.dumps({"complete": True, "counts": {"tables": 1},
                "files": {"catalog/tables.jsonl": {"sha256": sha256(table)}}}))
            index = BM25Index(docs, sha256(table))
            path = root / "index.json"
            index.save(path)
            self.assertEqual(BM25Index.load(path, root).search("apple")["results"], index.search("apple")["results"])
            with self.assertRaises(FileExistsError):
                index.save(path)
            table.write_text("corrupt")
            with self.assertRaises(ValueError):
                BM25Index.load(path, root)

    def test_evaluation_denominators_missing_tables_and_no_gold_query_leak(self):
        index = BM25Index([document("a", "apple"), document("b", "pear")])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dev.jsonl"
            path.write_text(json.dumps({"id": "q", "question": "apple", "db_id": "shop",
                                       "gold_document_ids": ["a", "b"]}) + "\n")
            report = evaluate(index, path, k=5, scoped=False)
            self.assertEqual(report["all"]["recall"], .5)
            self.assertEqual(report["all"]["precision_at_k"], .2)
            self.assertEqual(report["all"]["complete"], 0)


class APITests(unittest.TestCase):
    def test_authenticated_api_filters_scope_and_validates_body(self):
        index = BM25Index([document("a", "apple"), document("s", "apple", "private")])
        server = HTTPServer(("127.0.0.1", 0), make_handler(index, "t" * 32, {"shop"}))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(body, authenticated=True):
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                headers = {"Content-Type": "application/json"}
                if authenticated:
                    headers["Authorization"] = "Bearer " + "t" * 32
                connection.request("POST", "/search", json.dumps(body), headers)
                response = connection.getresponse()
                status, data = response.status, json.loads(response.read())
                connection.close()
                return status, data
            self.assertEqual(request({"query": "apple"}, False)[0], 401)
            status, data = request({"query": "apple"})
            self.assertEqual(status, 200)
            self.assertEqual([hit["document_id"] for hit in data["results"]], ["a"])
            self.assertEqual(request({"query": "apple", "db_id": "private"})[0], 400)
            self.assertEqual(request({"query": "apple", "allowed_db_ids": ["private"]})[0], 400)
            self.assertEqual(request({"query": "apple", "k": 100})[0], 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

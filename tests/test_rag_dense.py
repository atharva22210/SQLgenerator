"""Offline dense mathematics and integration; no pretrained weights downloaded."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rag.common.catalog import sha256
from rag.common.cli import main
from rag.common.evaluation import evaluate
from rag.dense import DenseIndex
from rag.dense.encoder import MiniLMEncoder, mean_pool


def document(identity, table, db="shop"):
    return {"document_id": identity, "db_id": db, "table": table, "domain": "",
            "domain_description": "", "columns": [], "primary_key": [], "foreign_keys": [],
            "text": table, "schema_sql": f"CREATE TABLE {table} (id INT)"}


class FixtureEncoder:
    """Fixed vectors let tests check ranking independently of model quality."""
    def __init__(self):
        self.seen = []

    def configuration(self):
        return {"model_id": "offline-fixture", "revision": "1"}

    def encode(self, texts, *, batch_size=32):
        self.seen.extend(texts)
        values = {"customers": [3, 4], "orders": [1, 0], "private": [1, 0], "buyers": [1, 0]}
        return np.array([values[text.strip()] for text in texts], dtype=np.float32)


class DenseTests(unittest.TestCase):
    def setUp(self):
        self.docs = [document("z", "private", "secret"), document("b", "orders"),
                     document("a", "orders"), document("c", "customers")]
        self.encoder = FixtureEncoder()
        self.index = DenseIndex(self.docs, self.encoder)

    def bundle(self, root):
        (root / "catalog").mkdir()
        path = root / "catalog/tables.jsonl"
        path.write_text("".join(json.dumps(doc) + "\n" for doc in self.docs))
        (root / "manifest.json").write_text(json.dumps({"complete": True, "counts": {"tables": len(self.docs)},
            "files": {"catalog/tables.jsonl": {"sha256": sha256(path)}}}))
        return sha256(path)

    def test_cosine_permissions_ties_and_empty_scope(self):
        response = self.index.search("buyers", k=3, allowed_db_ids={"shop"}, explain=True)
        self.assertEqual([r["document_id"] for r in response["results"]], ["a", "b", "c"])
        np.testing.assert_allclose([r["score"] for r in response["results"]], [1, 1, .6])
        self.assertEqual(response["candidate_count"], 3)
        self.assertNotIn("term_scores", response["results"][0])
        self.assertEqual(self.index.search("buyers", allowed_db_ids=set())["results"], [])
        for kwargs in ({"db_id": "secret", "allowed_db_ids": {"shop"}}, {"db_id": "unknown"},
                       {"k": True}, {"k": 0}, {"allowed_db_ids": {"unknown"}}):
            with self.assertRaises(ValueError):
                self.index.search("buyers", **kwargs)
        with self.assertRaises(ValueError):
            self.index.search(" ")

    def test_invalid_vectors_and_query_dimension(self):
        for vectors in ([[0, 0]] * 4, [[float("nan"), 1]] * 4, [[1, 2]], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                DenseIndex(self.docs, self.encoder, embeddings=vectors)
        with patch.object(self.encoder, "encode", return_value=np.ones((1, 3))):
            with self.assertRaises(ValueError):
                self.index.search("buyers")

    def test_faiss_matches_numpy_with_scoped_ids_ties_zero_and_negative_scores(self):
        docs = [document("private", "private", "secret"), document("b", "orders"),
                document("a", "orders"), document("c", "orders"), document("d", "orders")]
        index = DenseIndex(docs, FixtureEncoder(), embeddings=[[1, 0], [1, 0], [1, 0], [0, 1], [-1, 0]])
        self.assertEqual(index.vector_index.ntotal, len(docs))
        candidates = [1, 2, 3, 4]  # Deliberately not all global document IDs.
        scores = index.embeddings[candidates] @ np.array([1, 0], dtype=np.float32)
        expected = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), docs[pair[0]]["document_id"]))
        response = index.search("buyers", allowed_db_ids={"shop"})
        self.assertEqual([hit["document_id"] for hit in response["results"]],
                         [docs[position]["document_id"] for position, _ in expected])
        np.testing.assert_allclose([hit["score"] for hit in response["results"]],
                                   [score for _, score in expected], atol=1e-6)
        # FAISS's tie ordering is unspecified; keep our ID rule even at k=1.
        self.assertEqual(index.search("buyers", k=1, db_id="shop")["results"][0]["document_id"], "a")

    def test_snapshot_roundtrip_reuses_vectors_and_rejects_staleness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.index.source_sha256 = self.bundle(root)
            path = root / "index.json"
            self.index.save(path)
            fresh = FixtureEncoder()
            loaded = DenseIndex.load(path, root, encoder=fresh)
            self.assertEqual(fresh.seen, [])  # Loading must not re-encode schemas.
            self.assertEqual(loaded.search("buyers")["results"], self.index.search("buyers")["results"])
            with self.assertRaises(FileExistsError):
                self.index.save(path)
            payload = json.loads(path.read_text())
            payload["encoder"]["revision"] = "different"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                DenseIndex.load(path, root, encoder=fresh)
            (root / "catalog/tables.jsonl").write_text("corrupt")
            with self.assertRaises(ValueError):
                DenseIndex.load(path, root, encoder=fresh)

    def test_shared_evaluation_keeps_gold_out_of_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            labels = Path(temp) / "dev.jsonl"
            labels.write_text(json.dumps({"id": "q", "question": "buyers", "db_id": "shop",
                                          "gold_document_ids": ["a", "c"]}) + "\n")
            self.encoder.seen.clear()
            report = evaluate(self.index, labels, k=2, scoped=True)
            self.assertEqual(self.encoder.seen, ["buyers"])
            self.assertEqual(report["method"], "dense")
            self.assertEqual(report["all"]["recall"], .5)
            self.assertEqual(report["all"]["precision_at_k"], .5)

    def test_cli_build_search_evaluate_with_shared_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.bundle(root)
            (root / "labels").mkdir()
            (root / "labels/dev.jsonl").write_text(json.dumps({"id": "q", "question": "buyers",
                "db_id": "shop", "gold_document_ids": ["a"]}) + "\n")
            prefix = ["rag", "--bundle", str(root), "--index", str(root / "index.json")]
            def run(tail):
                output = io.StringIO()
                with patch("sys.argv", prefix + tail), redirect_stdout(output):
                    main(default_backend="dense")
                return json.loads(output.getvalue())
            with patch("rag.dense.encoder.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.MiniLMEncoder", return_value=FixtureEncoder()), \
                 patch("rag.dense.index.encoder_configuration", return_value=FixtureEncoder().configuration()):
                self.assertEqual(run(["build"])["tables"], 4)
                result = run(["search", "--query", "buyers", "--db-id", "shop", "--k", "1"])
                self.assertEqual(result["results"][0]["document_id"], "a")
                report = root / "report.json"
                self.assertEqual(run(["evaluate", "--scoped", "--k", "1", "--output", str(report)])
                                 ["all"]["recall"], 1)
                self.assertEqual(len(json.loads(report.read_text())["records"]), 1)


class EncoderTests(unittest.TestCase):
    def test_real_transformer_encoding_is_batch_invariant_and_frozen(self):
        from transformers import BertConfig, BertModel, BertTokenizerFast

        # Tiny random weights check inference plumbing, not semantic quality.
        with tempfile.TemporaryDirectory() as temp:
            vocab = Path(temp) / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbuyers\norders\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            model = BertModel(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16))
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("transformers.AutoModel.from_pretrained", return_value=model):
                encoder = MiniLMEncoder()
            texts = ["buyers", "buyers orders orders"]
            separate = encoder.encode(texts, batch_size=1)
            together = encoder.encode(texts, batch_size=2)
            self.assertEqual(together.shape, (2, 8))
            np.testing.assert_allclose(separate, together, atol=1e-6)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_pooling_excludes_padding(self):
        hidden = torch.tensor([[[2., 4.], [4., 8.], [100., 100.]]])
        pooled = mean_pool(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[3., 6.]]))

    def test_real_transformer_encode_batches_without_network(self):
        # A tiny randomly initialized BERT checks the actual HF/PyTorch path,
        # device, masking and batching; it says nothing about retrieval quality.
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=16)).eval()
        def tokenizer(texts, **kwargs):
            self.assertEqual(kwargs["max_length"], 256)
            return {"input_ids": torch.tensor([[1, 2, 0]] * len(texts)),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(texts))}
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("transformers.AutoModel.from_pretrained", return_value=model):
            encoder = MiniLMEncoder()
            vectors = encoder.encode(["a", "b", "c"], batch_size=2)
            self.assertEqual(vectors.shape, (3, 8))
            np.testing.assert_allclose(vectors[0], vectors[2], rtol=1e-5, atol=1e-6)
            self.assertTrue(np.isfinite(vectors).all())
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

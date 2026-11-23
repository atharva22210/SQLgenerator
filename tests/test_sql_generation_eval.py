"""Check prompt isolation, decoding boundaries, and a real CPU checkpoint round trip."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.transformer_35m import Transformer, TransformerConfig
from pretraining_experiments.sql_generation_eval.evaluate import evaluate, greedy_completion, split_document


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class GenerationTests(unittest.TestCase):
    def test_reference_and_ambiguous_delimiters(self):
        prompt = 'Schema:\nt(a)\n\nAdditional context:\na is age\n\nQuestion:\nOldest age?\n<sql>\n'
        self.assertEqual(split_document(prompt + 'SELECT MAX(a) FROM t\n</sql>'),
                         (prompt, 'SELECT MAX(a) FROM t'))
        self.assertIsNone(split_document(prompt + 'SELECT\n<sql>\nsecret\n</sql>'))
        self.assertIsNone(split_document('CREATE TABLE t(a);'))

    def test_multitoken_tag_eos_and_length_cap(self):
        class Script:
            def __init__(self, answer):
                self.ids = iter(map(ord, answer))

            def generate(self, context, max_new_tokens):
                return torch.cat([context, torch.tensor([[next(self.ids)]])], dim=1)

        for text, expected, reason in [('X</sql>extra', 'X', 'closing_tag'), ('X\0', 'X', 'eos'),
                                       ('123456789', '12345678', 'token_limit')]:
            row = greedy_completion(Script(text), CharacterTokenizer(), [80], 8, 'cpu')
            self.assertEqual((row['generated_sql'], row['stop_reason']), (expected, reason))

    def test_final_checkpoint_generates_repeatable_held_out_rows(self):
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = ['SELECT 0', *[f'Schema:\nt(a)\n\nQuestion:\nQ{i}?\n<sql>\nSELECT {i}\n</sql>'
                                  for i in range(5)]]
            docs.append('Schema:\n' + 't' * 120 + '\n\nQuestion:\nQ?\n<sql>\nSELECT long\n</sql>')
            np.array(tokenizer.encode('\0'.join(docs) + '\0'), dtype='<u4').tofile(root / 'sql_val.bin')
            identity = dict(sql_dir=tmp, tokenizer_sha256='abc', corpus_id='test', manifest_sha256='xyz')
            config = TransformerConfig(vocab_size=128, d_model=16, n_heads=2, n_kv_heads=1,
                                       n_layers=1, ffn_dim=32, max_seq_len=96)
            state = dict(model_config=asdict(config), model=Transformer(config).state_dict(),
                         config={'max_steps': 5}, update=5, grouped_sql_dir=tmp,
                         dataset_identity=identity, wandb_id='test')
            path = root / 'latest.pt'
            torch.save(state, path)
            module = 'pretraining_experiments.sql_generation_eval.evaluate'
            with patch(module + '.dataset_identity', return_value=identity), patch(
                    module + '.tokenizer_fingerprint', return_value='abc'):
                args = (path, root, tokenizer, 3, 12345, 2, 'cpu')
                rows = evaluate(*args)
                self.assertEqual(rows, evaluate(*args))
                self.assertEqual(len({r['validation_document'] for r in rows}), 3)
                for row in rows:
                    self.assertIn(row['validation_document'], range(1, 6))
                    self.assertNotIn('SELECT', row['prompt'])
                    self.assertEqual(row['reference_sql'], f"SELECT {row['validation_document'] - 1}")
                with self.assertRaisesRegex(ValueError, 'only 5 eligible'):
                    evaluate(path, root, tokenizer, 6, 12345, 2, 'cpu')
                torch.save({**state, 'update': 4}, path)
                with self.assertRaisesRegex(ValueError, 'final checkpoint'):
                    evaluate(*args)

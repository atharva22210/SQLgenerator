"""Check that formatting differences cannot put the same task in validation."""
import unittest
from data.sql_overlap import task_key


class SQLOverlapTests(unittest.TestCase):
    def test_cross_source_formatting(self):
        first = '-- SQL task\n-- How many users?\n<sql>SELECT COUNT(*) FROM users</sql>'
        second = 'Schema:\nCREATE TABLE users(id int)\nQuestion:\nHow many users?\n<sql>select  COUNT(*) from users</sql>'
        self.assertEqual(task_key(first, 'nstext2sql'), task_key(second, 'sql_context'))

    def test_different_answers(self):
        text = 'Question:\nHow many?\n<sql>SELECT COUNT(*) FROM {}</sql>'
        self.assertNotEqual(task_key(text.format('users'), 'sql_context'),
                            task_key(text.format('orders'), 'sql_context'))

    def test_unknown_format_fails(self):
        with self.assertRaises(ValueError):
            task_key('unrecognized document', 'sql_context')

    def test_filter_excludes_training_and_validation_copies(self):
        import numpy as np
        from types import SimpleNamespace
        from data.sql_overlap import filter_validation

        def document(question, source):
            prompt = f'-- {question}' if source == 0 else f'Question:\n{question}'
            return prompt + '\n<sql>SELECT 1</sql>'

        def file(texts):
            return SimpleNamespace(tokens=np.array([ord(c) for t in texts for c in t]),
                                   ends=np.cumsum([len(t) for t in texts]) - 1)

        train = SimpleNamespace(sql=[file([document('seen', i)]) for i in range(3)])
        validation = SimpleNamespace(sql=[file([document('seen', i), document('new', i)])
                                          for i in range(3)])
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ''.join(map(chr, ids)))
        report = filter_validation(train, validation, tokenizer)
        self.assertEqual(validation.sql[0].eligible_indices.tolist(), [1])
        self.assertEqual(validation.sql[1].eligible_indices.tolist(), [])
        self.assertEqual(report['sql_context']['repeated_validation'], 1)
        self.assertEqual(validation.sql_weights[1:], [0, 0])

    def test_filtered_sampling_preserves_original_document_boundaries(self):
        import random
        import tempfile
        from pathlib import Path
        import numpy as np
        from data.dataloader import DocumentFile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'val.bin'
            np.array([10, 99, 20, 21, 99, 30, 99], dtype='<u4').tofile(path)
            source = DocumentFile(path, eos_id=99)
            source.eligible_indices = np.array([1], dtype=np.int64)
            self.assertEqual(source.sample(random.Random(0), 20), [20, 21, 99])
            source.eligible_indices = np.array([], dtype=np.int64)
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                source.sample(random.Random(0), 20)

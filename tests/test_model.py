import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from sqlgenerator.model import CausalLanguageModel, ModelConfig


@unittest.skipUnless(torch is not None, "PyTorch is required")
class CausalLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CausalLanguageModel(
            ModelConfig(
                vocab_size=17,
                hidden_size=16,
                num_layers=1,
                num_heads=4,
                max_sequence_length=8,
                dropout=0,
            )
        )

    def test_forward_returns_logits_and_shifted_loss(self) -> None:
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

        logits, loss = self.model(tokens, labels=tokens)

        self.assertEqual(logits.shape, (2, 3, 17))
        self.assertTrue(torch.isfinite(loss))

    def test_generate_appends_requested_tokens(self) -> None:
        generated = self.model.generate(torch.tensor([[1, 2]]), max_new_tokens=3)

        self.assertEqual(generated.shape, (1, 5))


class ModelConfigTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_rejects_incompatible_attention_heads(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(hidden_size=10, num_heads=3)

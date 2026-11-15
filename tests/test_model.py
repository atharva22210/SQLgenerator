"""Small behavioral checks for the transformer; no GPU or dataset is needed."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.transformer_35m import Transformer, TransformerConfig


@unittest.skipUnless(torch is not None, "PyTorch is required")
class TransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Tiny dimensions make the actual model fast enough to inspect on CPU.
        self.config = TransformerConfig(
            vocab_size=19, d_model=16, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_dim=32, max_seq_len=8,
        )
        self.model = Transformer(self.config)

    def test_forward_loss_and_backward(self) -> None:
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        logits, loss = self.model(tokens, labels=tokens)

        self.assertEqual(logits.shape, (2, 3, 19))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)
        self.assertTrue(torch.isfinite(self.model.token_embedding.weight.grad).all())

    def test_chunked_loss_matches_full_loss_and_embedding_gradient(self) -> None:
        """Small vocabulary makes both paths cheap enough to compare exactly."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, -100, 4, 5]])
        _, full_loss = self.model(tokens, labels=labels)
        full_loss.backward()
        full_gradient = self.model.token_embedding.weight.grad.clone()
        self.model.zero_grad(set_to_none=True)

        logits, chunked_loss = self.model(tokens, labels=labels, loss_chunk_size=2)
        self.assertIsNone(logits)
        chunked_loss.backward()
        torch.testing.assert_close(chunked_loss, full_loss)
        torch.testing.assert_close(self.model.token_embedding.weight.grad, full_gradient)

    def test_future_tokens_cannot_change_earlier_predictions(self) -> None:
        self.model.eval()
        first, _ = self.model(torch.tensor([[1, 2, 3, 4]]))
        changed_future, _ = self.model(torch.tensor([[1, 2, 8, 9]]))

        # If causality is broken, changing tokens 3 and 4 changes the outputs
        # for earlier tokens 1 and 2 as well.
        torch.testing.assert_close(first[:, :2], changed_future[:, :2])

    def test_document_mask_isolates_packed_examples(self) -> None:
        self.model.eval()
        documents = torch.tensor([[0, 0, 1, 1]])
        same_document = documents[:, :, None] == documents[:, None, :]
        positions = torch.tensor([[0, 1, 0, 1]])
        first, _ = self.model(
            torch.tensor([[1, 2, 3, 4]]),
            position_ids=positions, attention_mask=same_document,
        )
        changed_first_document, _ = self.model(
            torch.tensor([[7, 8, 3, 4]]),
            position_ids=positions, attention_mask=same_document,
        )
        torch.testing.assert_close(first[:, 2:], changed_first_document[:, 2:])

    def test_qk_norm_and_greedy_generation(self) -> None:
        config = TransformerConfig(
            vocab_size=19, d_model=16, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_dim=32, max_seq_len=8, use_qk_norm=True,
        )
        model = Transformer(config)
        generated = model.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
        self.assertEqual(generated.shape, (1, 5))
        torch.testing.assert_close(generated[:, :2], torch.tensor([[1, 2]]))

    def test_rejects_incompatible_attention_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerConfig(vocab_size=19, d_model=10, n_heads=3)


if __name__ == "__main__":
    unittest.main()

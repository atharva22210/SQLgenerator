"""Small behavioral checks for the transformer; no GPU or dataset is needed."""

import unittest
import random
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from model.transformer_35m import Transformer, TransformerConfig
    from data.dataloader import make_batch


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

    def test_layout_matches_reference_loss_and_all_parameter_gradients(self) -> None:
        """Compare new metadata against the original document-mask semantics."""
        class Corpus:
            calls = 0

            def sample(self, kind, rng, length, *, first_slice):
                self.calls += 1
                return ([1, 2, 3] if self.calls % 2 else [4, 5])[:length]

        for mode in ("pad", "naive_pack", "masked_pack"):
            with self.subTest(mode=mode):
                batch = make_batch(Corpus(), mode=mode, batch_size=2, seq_len=8,
                                   sql_fraction=0.5, pad_id=0, rng=random.Random(0))
                self.model.zero_grad(set_to_none=True)
                mask = None
                if mode == "masked_pack":
                    ids = batch.document_ids
                    mask = ids[:, :, None] == ids[:, None, :]
                _, reference = self.model(batch.input_ids, labels=batch.labels,
                                          position_ids=batch.position_ids, attention_mask=mask)
                reference.backward()
                gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()}
                self.model.zero_grad(set_to_none=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    _, actual = self.model(**batch.model_inputs(mode, torch.device("cpu")),
                                           loss_chunk_size=3)
                    actual.backward()
                # The trusted CPU count must avoid .item(), bool(tensor), or
                # int(tensor) reads in the training forward/backward path.
                self.assertNotIn("aten::_local_scalar_dense", {event.key for event in trace.key_averages()})
                torch.testing.assert_close(actual, reference)
                for name, parameter in self.model.named_parameters():
                    torch.testing.assert_close(parameter.grad, gradients[name])

    def test_cached_rope_and_causal_mask_are_shared_across_layers(self) -> None:
        from model.transformer_35m import apply_rope

        model = Transformer(TransformerConfig(
            vocab_size=19, d_model=16, n_layers=3, n_heads=4,
            n_kv_heads=2, ffn_dim=32, max_seq_len=8,
        ))
        positions = torch.tensor([[0, 1, 0, 1]])
        x = torch.randn(1, 4, 4, 4)
        frequencies = model.config.rope_theta ** (-torch.arange(0, 4, 2).float() / 4)
        angles = positions.float()[..., None] * frequencies
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., ::2], x[..., 1::2]
        expected = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)
        torch.testing.assert_close(apply_rope(x, model.rope_cosine[positions][:, None],
                                             model.rope_sine[positions][:, None]), expected)
        seen = []
        handles = [block.register_forward_pre_hook(lambda module, args: seen.append(args[1:3]))
                   for block in model.blocks]
        try:
            with patch.object(torch.Tensor, "cos", side_effect=AssertionError("recomputed cos")), \
                 patch.object(torch.Tensor, "sin", side_effect=AssertionError("recomputed sin")):
                model(torch.tensor([[1, 2, 3, 4]]), position_ids=positions,
                      attention_mask=torch.ones(1, 4, 4, dtype=torch.bool))
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(all(rotations is seen[0][0] and allowed is seen[0][1]
                            for rotations, allowed in seen))
        self.assertFalse(any("rope" in key for key in model.state_dict()))

    def test_chunked_loss_rejects_all_ignored_targets_for_untrusted_callers(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmasked"):
            self.model(torch.tensor([[1, 2]]), labels=torch.tensor([[-100, -100]]),
                       loss_chunk_size=1)

    def test_accumulation_weights_by_targets_with_unequal_padding(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, 6, 7]])
        _, reference = self.model(tokens, labels=labels)
        reference.backward()
        gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        counts = [2, 3]
        for row, count in enumerate(counts):
            _, loss = self.model(tokens[row:row + 1], labels=labels[row:row + 1],
                                 loss_chunk_size=2, valid_target_count=count)
            (loss * (count / sum(counts))).backward()
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name])

    @unittest.skipUnless(torch is not None and torch.cuda.is_available()
                         and torch.cuda.is_bf16_supported(), "CUDA bf16 GPU required")
    def test_cuda_varlen_matches_dense_reference_in_every_mode(self) -> None:
        """Run on an A10 before training: exercise real FA2 forward/backward/GQA.

        bf16 fused attention has a different reduction order from dense math,
        so comparisons allow bf16 rounding rather than requiring exact bits.
        """
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from model.transformer_35m import AttentionLayout

        class Corpus:
            calls = 0

            def sample(self, kind, rng, length, *, first_slice):
                self.calls += 1
                return ([1, 2, 3] if self.calls % 2 else [4, 5])[:length]

        model = Transformer(TransformerConfig(
            vocab_size=19, d_model=128, n_layers=2, n_heads=2,
            n_kv_heads=1, ffn_dim=192, max_seq_len=8, use_qk_norm=True,
        )).cuda()
        for mode in ("pad", "naive_pack", "masked_pack"):
            with self.subTest(mode=mode):
                batch = make_batch(Corpus(), mode=mode, batch_size=2, seq_len=8,
                                   sql_fraction=0.5, pad_id=0, rng=random.Random(0))
                inputs = batch.model_inputs(mode, torch.device("cuda"))
                mask = inputs["attention_layout"].dense_mask(2, 8)
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.MATH):
                    _, expected = model(inputs["input_ids"], labels=inputs["labels"],
                                        position_ids=inputs["position_ids"], attention_mask=mask)
                expected.backward()
                gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                with patch.object(AttentionLayout, "dense_mask", side_effect=AssertionError("dense CUDA mask")), \
                     torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, actual = model(**inputs, loss_chunk_size=3)
                    actual.backward()
                self.assertNotIn("aten::_local_scalar_dense", {e.key for e in trace.key_averages()})
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=8e-2, atol=5e-3)

    def test_rejects_incompatible_attention_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerConfig(vocab_size=19, d_model=10, n_heads=3)


if __name__ == "__main__":
    unittest.main()

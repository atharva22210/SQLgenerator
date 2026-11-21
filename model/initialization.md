# Initialization for our SwiGLU transformer

Fresh models sample linear and token embedding weights from a zero-mean normal
distribution with standard deviation `initializer_range`, defaulting to `0.02`.
RMSNorm scales start at one. Vocabulary output shares the token embedding, so
there is only one learned matrix to initialize for those two uses.

This follows a concrete implementation: Hugging Face Transformers v4.57.1's
[Llama feed-forward layer](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)
computes `down(silu(gate(x)) * up(x))`, as ours does. Its
[configuration](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/configuration_llama.py)
defaults `initializer_range` to `0.02`, and the inherited
[initializer](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py)
uses `normal_(mean=0, std=initializer_range)` for linear and embedding weights.
The executable rule is an untruncated normal. This reference does not establish
the training recipe used for Meta's released weights, or an optimum for our model.

## Where does the 1/n term fit?

If a projection sums `n` approximately independent inputs with unit variance,
its output variance is approximately `n * Var(weight)`. Fan-in initialization
therefore chooses weight variance proportional to `1/n`, which means standard
deviation proportional to `1/sqrt(n)`. `n` is the input width of that projection:
512 for our gate/up layers, 1408 for our down layer. It is not the layer count.

This principle still applies to projections inside SwiGLU. However, SwiGLU
multiplies two branches, one with SiLU, so the gain derived for a single ReLU
does not automatically preserve the variance of the entire feed-forward block.
There is no single initializer required by the SwiGLU activation. For example,
[OLMo's configuration](https://github.com/allenai/OLMo/blob/main/olmo/config.py)
supports both fixed-normal initialization and a fan-in option with standard
deviation `1/sqrt(d_in)`.

Our selected fixed-normal rule has **no additional fan-in division**. Its
standard deviation is `0.02`, not `0.02/sqrt(n)`. Changing the model width does
not automatically change this value. ReLU He initialization instead uses
`sqrt(2/n)`; it is not the rule selected here. Depth-based residual scaling is
another choice and is not applied. Attention's existing `1/sqrt(head_dim)`
score scaling is independent of weight initialization.

## Configuration and saved runs

```python
config = TransformerConfig(vocab_size=32768, initializer_range=0.02)
model = Transformer(config)
```

Both the packing and distributed Modal launchers accept `--initializer-range`.
The default preserves the original initialization exactly for a fixed seed.
Every distributed case and rank receives the same setting from `TrainConfig`;
architecture overrides do not replace that training setting. No dependencies
or remote image changes are needed; use the project's locked Python environment.

The setting is recorded with training configuration, including W&B and
checkpoints. Packing run names generated for a nondefault scale include an
`-init-std...` suffix; explicit run names must still identify distinct experiments.
Loading a checkpoint restores its learned values rather than reinitializing
them. Packing resume requires matching configuration: an older checkpoint
without this field means `0.02`; other missing or different fields remain errors.

## Validation and future comparisons

```bash
uv run --locked python -m unittest discover -s tests -p 'test_weight_initialization.py'
uv run --locked python -m unittest discover -s tests
```

CPU checks cover initialization scale, equality with the HF Llama rule, exact
baseline parity, finite forward/backward results, checkpoint restoration,
legacy resume rules and launcher propagation. Mocked submissions allocate no
GPUs. No training experiment is part of this change.

Before comparing initializers in training, agree on the hypothesis and predict
activation/gradient magnitudes and initial loss. Keep architecture, seed, data,
token budget and optimizer fixed. Compare nonfinite values, gradient norms,
clipping frequency and held-out loss in a short single-GPU run, with its step
count and resource budget chosen before launch. Correctness tests alone do not
establish which initializer trains better.

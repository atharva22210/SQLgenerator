# Hybrid candidates, then cross-encoder reranking

This version reranks a small plain-hybrid candidate pool, independently of
graph search and column selection. Read `model.py`, then `index.py`.

```text
question -> hybrid top 8 permitted tables
                       -> (question, table 1 text) -> score
                       -> (question, table 2 text) -> score
                       -> ...
                       -> sort scores -> final top 4 schemas
```

Dense retrieval embeds question and table separately. A cross-encoder feeds
each question/table pair through the same transformer input, allowing attention
between them. Each candidate gets its own pair; we do not put all candidate
tables together in one model input. Scores are raw relevance logits, may be
negative, and are not calibrated probabilities or comparable to RRF scores.

We use frozen
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
pinned to revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, following its
published Transformers sequence-classification recipe. It was trained for
passage ranking, not our SQL schemas; any relevance improvement is a hypothesis.
Joint inputs are limited to 512 tokens including special tokens; long schemas
or questions can be truncated. Evaluation mode, frozen parameters and inference
mode disable training. No new dependency is needed beyond locked Transformers
and PyTorch, but first build/load may download a second model (about 91 MB of
weights plus tokenizer files). CPU is the default; no remote GPU is allocated.

The cross-encoder scores exactly the existing schema-only `searchable_text()`
representation, never SQL answers or gold table IDs. Permissions and database
scope apply inside hybrid BEFORE candidate selection. Reranking cannot recover
tables absent from that candidate pool, so a smaller pool bounds latency but
can reduce recall. Default pool size is eight; it is configurable from 1..20.
Final k defaults to four and must not exceed the configured pool size. A
database with fewer tables simply returns fewer candidates. Ties use IDs.

Each result's `score` is replaced with the cross-encoder logit, while
`hybrid_score` preserves the first-stage score. `--explain` adds hybrid rank
and preserves its explanation under `hybrid_explanation` so word/rank
contributions are not mistaken for the new score. `rerank_candidate_count`
counts scored pairs. Final table schemas are not reduced by this backend.

## Commands

Use the project's Python 3.13 and `uv sync --locked` environment:

```bash
uv run --locked python -m rag.hybrid_rerank.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-rerank-v1/index.json build --candidate-pool 8

uv run --locked python -m rag.hybrid_rerank.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-rerank-v1/index.json search \
  --query 'Which customers placed the largest orders?' --k 4 --explain

uv run --locked python -m rag.hybrid_rerank.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-rerank-v1/index.json evaluate \
  --split dev --k 4 --output artifacts/rag/hybrid-rerank-v1/dev-global.json
```

The shared CLI/API accept `--backend hybrid_rerank`. Snapshot JSON stores the
verified nested hybrid index, candidate budget and pinned reranker identity,
not duplicate model weights. Loading rejects a changed recipe or catalogue.
The shared evaluator scores the FINAL reranked results, not the original pool.

## What to evaluate next

Question: does pairwise scoring improve top-4 table selection enough to justify
its latency? Hypothesis: question/table interaction resolves ambiguous schemas.
Baseline: plain hybrid, final k=4. Variable: cross-encoder reranking of a fixed
eight-candidate pool. Keep catalogue, dev labels, scope and all other retrieval
settings fixed. Compare recall, complete coverage, precision, MRR and p50/p95
retrieval latency; inspect the multi-table subset too. Candidate recall is an
upper bound on what this reranker can recover. Expected resources: local CPU
plus up to eight pair forward passes per query; no training or generation.

Offline tests use fixed scores to verify rank changes and final-order evaluation.
A tiny random local classifier checks real paired token inputs, batching and
frozen inference, without downloading the pretrained model. No quality/latency
benchmark or SQL execution accuracy is claimed by these correctness tests.

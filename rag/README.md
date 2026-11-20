# Schema retrieval and hybrid refinements

All methods retrieve table schemas, not SQL answers. They read the same
verified `catalog/tables.jsonl`; `databases.jsonl` and `relationships.jsonl`
are not loaded by search. Foreign-key names already present in table records
contribute to the shared searchable text. Labels stay separate until scoring.

```text
rag/
  bm25/
    index.py          # Word postings and BM25 term contributions
    cli.py            # python -m rag.bm25.cli
  dense/
    encoder.py        # Frozen transformer: strings -> vectors [B,384]
    index.py          # Document vectors, FAISS cosine search, snapshot checks
    faiss_search.py   # Exact search; isolates FAISS from torch on macOS
    cli.py            # python -m rag.dense.cli
  hybrid/
    index.py          # Combine BM25 and dense rankings with rank fusion
    cli.py            # python -m rag.hybrid.cli
  hybrid_graph/
    graph.py          # Question-dependent edge costs and weighted Dijkstra
    connections.py    # Kruskal selection and cycle/bridge-leaf pruning
    index.py          # Four hybrid seeds plus query-weighted connecting paths
    cli.py            # python -m rag.hybrid_graph.cli
  hybrid_columns/
    selection.py      # Column descriptions, mandatory keys, reduced DDL
    index.py          # Hybrid tables -> column scores -> reduced context
    cli.py            # python -m rag.hybrid_columns.cli
  hybrid_rerank/
    model.py          # Joint question/table scoring with a frozen cross-encoder
    index.py          # Small hybrid pool -> rerank -> top-k
    cli.py            # python -m rag.hybrid_rerank.cli
  common/
    catalog.py        # File checksum verification and schema-only search text
    prepare_data.py   # Shared catalogue and separate train/dev/test labels
    search.py         # by_db, request validation, permission filter, result fields
    storage.py        # Atomic JSON publication without overwriting
    evaluation.py     # Shared retrieval metrics
    cli.py            # Shared argument parsing and command handling
    service.py        # Shared authenticated local HTTP API
  cli.py              # Existing command still defaults to BM25
  prepare_data.py     # Existing data-preparation entry point/imports
  service.py          # Existing local API entry point/imports
```

Read `common/catalog.py`, then the BM25/dense indexes and `hybrid/index.py`, then
`common/cli.py` and `common/evaluation.py`. Existing `from rag.bm25 import
BM25Index` imports and version-1 BM25 snapshots remain supported.

For hybrid seeds followed by question-dependent FK path expansion, see
[the graph walkthrough](hybrid_graph/README.md). It keeps the existing hybrid
backend separate and adds `hybrid_graph` to the shared CLI and local API.

Two independent refinements start from plain hybrid:
[table-to-column selection](hybrid_columns/README.md) keeps semantic columns
plus mandatory join keys; [cross-encoder reranking](hybrid_rerank/README.md)
scores a small candidate pool jointly with the question. They do not run graph
expansion or automatically compose with each other. Both use the common
evaluation and API, with k=4 by default.

## Dense data flow

1. Verify the catalogue against its manifest, as BM25 does.
2. Render each table's searchable fields using the same `searchable_text()`.
3. Encode schema text in batches using a frozen pretrained transformer.
   Its output has shape `[batch, tokens, hidden]`. Average token vectors using
   the attention mask so padding does not affect the sentence vector.
4. Normalize each document vector to length one and save the `[tables,384]`
   matrix with its catalogue checksum and encoder configuration. Add the vectors
   to a CPU FAISS `IndexFlatIP` in the same document order.
5. At search time, use `by_db` to find permitted document positions, encode
   the question, normalize it, and ask FAISS to score only those vector IDs.
   For unit vectors, `document_vector @ query_vector` equals cosine similarity.
6. Sort eligible documents by score and return up to k schemas. Equal scores
   use document IDs for deterministic ordering.

The encoder is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Mean pooling
follows its published Transformers recipe. The token limit is 256 wordpieces,
including special tokens; wide schemas can lose trailing columns. Documents
and questions use the same encoding recipe. Snapshot loading rejects a
different catalogue, encoder configuration, or representation version.

This is a pretrained retrieval baseline; it neither trains the encoder nor
changes the from-scratch SQL generator. It uses Transformers, PyTorch, NumPy,
and `faiss-cpu` (locked to 1.13.2). Run `uv sync --locked` locally after updating;
remote environments must sync the updated lockfile too. FAISS scoring runs on
CPU even if the encoder uses a GPU. BM25 does not import FAISS.

FAISS replaces the NumPy matrix multiplication inside dense search; hybrid
uses that same dense object automatically. `IndexFlatIP` is exact inner-product
search, equivalent to cosine for our normalized vectors. It scans vectors;
there is no clustering, hierarchy, index training or approximation in this
milestone. See the [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

The index is built once per loaded `DenseIndex`. A per-query ID selector applies
the database scope inside FAISS, before ranking. We retrieve all eligible scores
and retain Python sorting so tied documents keep our document-ID ordering even
at the top-k boundary. This deliberately small integration does not promise a
speedup; it still sorts all eligible hits. A later throughput experiment could
compare native top-k or approximate indexes against this exact reference.
On macOS, PyTorch and FAISS wheels can load conflicting OpenMP runtimes.
`faiss_search.py` therefore keeps FAISS in a persistent subprocess on macOS,
loading the vectors once and sending queries/eligible IDs as JSON. Linux calls
FAISS directly. macOS has process and serialization overhead; no unsafe
`KMP_DUPLICATE_LIB_OK` workaround is used.
The existing JSON snapshots still store vectors, not FAISS binary files; the
in-memory FAISS index rebuilds on load, including from older dense snapshots.

Dense ranks every eligible table, including zero or negative cosine scores.
Unlike BM25's `no_lexical_match`, it has no lexical-match requirement and no
calibrated confidence threshold. An empty allowed scope returns `no_candidates`.
`--explain` describes cosine scoring and encoder identity; it does not assign
word contributions. Scores are not comparable numerically to BM25 scores.

## Run locally

Use the repository's Python 3.13 environment with `uv sync --locked`. CPU is
the default; first dense use downloads tokenizer files and roughly 91 MB of
model weights into the local Hugging Face cache. No remote GPU is allocated.
Use `--device cuda` before the subcommand only in a configured GPU environment.
BM25 commands never load Transformers or download a model.

From the repository root, build a new dense snapshot:

```bash
uv run --locked python -m rag.dense.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/dense-v1/index.json build --batch-size 32
```

Search it:

```bash
uv run --locked python -m rag.dense.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/dense-v1/index.json search \
  --query 'Which customers placed orders?' --k 5 --explain
```

For BM25, use `python -m rag.bm25.cli` and a separate BM25 index path.
The existing `python -m rag.cli` also accepts `--backend dense` before the
subcommand. Both expose the same search/evaluation arguments. Existing index
and evaluation report paths cannot be overwritten; choose a new version.

The API supports dense with `python -m rag.service --backend dense` plus its
existing bundle/index/allow-db arguments. Credentials and server-side database
permissions follow the [BM25 service instructions](../docs/rag-v1-bm25.md).
Clients cannot expand the server's allowlist. CLI use is trusted local access.

## Correctness and a later comparison

Run offline tests (no model download or retrieval benchmark):

```bash
uv run --locked python -m unittest discover -s tests -p 'test_rag*.py'
```

The dense tests check hand-calculated cosine scores, filtering before top-k,
ties, malformed vectors, stale snapshots, reuse of saved embeddings, CLI
integration and shared evaluation without leaking gold table IDs into the
encoder. A tiny randomly initialized local Transformer checks batching,
padding and frozen inference; it does not establish semantic retrieval quality.

Before a development comparison, predict whether the semantic baseline will
recover paraphrases better than BM25 and whether truncation will hurt wide
tables. Baseline: existing BM25 defaults. Variable: retrieval method only.
Keep catalogue, dev labels, k=5 and database scope fixed. Compare recall@5,
precision@5, complete coverage, reciprocal rank and p50/p95 retrieval latency,
including the multi-table subset. Expected resources: local CPU inference,
one model download, and document vectors using `tables * 384 * 4` bytes per
copy in memory (NumPy and FAISS each hold a copy). JSON snapshots take more
disk space; exact search scans every eligible
vector. This is deliberately a small baseline with no approximate index.

After discussing predictions, run the same evaluation command for each method
with different index/report paths:

```bash
uv run --locked python -m rag.dense.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/dense-v1/index.json evaluate \
  --split dev --k 5 --output artifacts/rag/dense-v1/dev-global.json
```

Add `--scoped` to supply the known database, treating that as a separate
comparison. Reserve test labels until choices are fixed. No dense benchmark
or claim of improvement accompanies this implementation.

## Hybrid data flow and commands

`HybridIndex` holds the existing BM25 and dense objects built from the same
ordered catalogue and checksum. Each child receives the same database scope
and server-controlled allowlist before ranking. Hybrid takes up to 20 results
from each child, fuses their union, and returns the final top k (1..20).
Keeping the child pool at 20 even when k=1 allows lower-ranked agreement to
change the winner. Tables outside both pools cannot appear in the answer.

We use equal-weight
[reciprocal rank fusion (RRF)](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).
For each document, starting ranks at 1:

```text
score = sum over child lists containing the document: 1 / (rrf_constant + rank)
```

With the default constant 60, a table at BM25 rank 2 and dense rank 1 gets
`1/62 + 1/61`. A table absent from BM25 receives only its dense contribution.
Raw BM25 and cosine scores are not added because they have different scales.
Fusion ties use document IDs. Scores remain ranking signals, not confidence.
Dense can supply low-similarity tables even when there are no word matches.

Build and save a self-contained hybrid snapshot:

```bash
uv run --locked python -m rag.hybrid.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-v1/index.json build \
  --batch-size 32 --k1 1.2 --b 0.75 --rrf-constant 60
```

The same frozen MiniLM encoder is used; first use may download its weights.
CPU is the default. Hybrid reuses dense's FAISS dependency and introduces no training. The snapshot
contains dense vectors and their catalogue/encoder identity plus the BM25
settings, analyzer version, fusion constant and child depth. Loading reuses
vectors and reconstructs postings from the verified documents. Existing index
paths are never overwritten; use a new path when changing settings.

```bash
uv run --locked python -m rag.hybrid.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-v1/index.json search \
  --query 'Which customers placed orders?' --k 5 --explain
```

`--explain` includes each child's rank, raw score, and RRF contribution for a
returned document. Adding the contributions reproduces its final score.
`candidate_count` counts all permitted schemas, while `matched_documents`
counts the union of child top-20 results before truncation to the final k.
Evaluation uses the same labels and metrics as the standalone methods:

```bash
uv run --locked python -m rag.hybrid.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-v1/index.json evaluate \
  --split dev --k 5 --output artifacts/rag/hybrid-v1/dev-global.json
```

Alternatively select `--backend hybrid` before the subcommand in `rag.cli`.
The local API also accepts `--backend hybrid` with the hybrid index path,
using the same token and `--allow-db` setup documented for BM25.

Before running a development experiment, predict whether combining lexical
matches with semantic similarity improves complete table coverage. Compare
against both standalone methods using the same catalogue, dev questions,
k=5 and scope; vary only the retrieval method. Record recall, precision,
complete coverage, reciprocal rank, multi-table coverage, and latency. Hybrid
adds BM25 search and fusion to the existing local dense inference workload;
no training or remote GPU allocation is needed. Correctness tests use fixed
vectors, not a benchmark; no hybrid quality improvement has been measured.

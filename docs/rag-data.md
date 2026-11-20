# RAG data: a schema catalogue and separate retrieval labels

## What was collected

We reuse the team's existing, completed Day 4 ingestion of
[Gretel synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
at revision `740ab236e64503fba51be1101df7a1be83bf455d` (Apache-2.0).
No new generator training or tokenizer preparation is needed for this step.

The existing `post-train-data` Modal Volume contains 45,000 SFT, 2,000 RL and
500 source-test examples. RL and source-test examples have retained MySQL
databases; SFT examples used temporary databases that were removed. Therefore
only RL/test schemas become searchable live-database documents. SFT questions
are still used to check evaluation overlap.

The collected bundle contains:

| Item | Count |
| --- | ---: |
| Existing MySQL databases | 2,500 |
| Table documents | 3,119 |
| Databases with multiple tables | 550 |
| Explicitly declared foreign keys | 17 |
| Retriever-training tasks from RL | 1,999 |
| Development tasks from source test | 250 |
| Held-out retrieval-test tasks from source test | 250 |
| Excluded tasks requiring no physical table | 1 |

"Retriever-training" means question/table labels available for a future
learned retriever. BM25/dense indexing does not need to train on these labels.
No model was trained by collection.

## Storage and what each file means

The shared bundle is on **post-train-data**, environment **main**, workspace
**sanika-05**, at `rag/gretel-v1-20b6e311a001/`. When mounted at `/vol`, its path
is `/vol/rag/gretel-v1-20b6e311a001/`.

The local copy is under the ignored
`artifacts/rag/gretel-v1-20b6e311a001/` directory. Raw source Parquet downloads
are in `artifacts/rag/source/`. Generated data stays out of Git; preparation
code, tests and this explanation are versioned.

```text
gretel-v1-20b6e311a001/
    catalog/
        databases.jsonl       database/domain descriptions and table IDs
        tables.jsonl          columns, types, keys, DDL and searchable text
        relationships.jsonl   only declared foreign-key relationships
    labels/
        train.jsonl           questions + gold tables + reference SQL
        dev.jsonl             tune retrieval settings using these tasks
        test.jsonl            reserve until the settings are fixed
        excluded.jsonl        rejected task IDs and reasons
    samples.md                readable single-table/multi-table examples
    source_manifest.json      original ingestion provenance
    manifest.json             counts, checksums, limits and completion marker
```

**Index only `catalog/`.** Do not recursively embed the whole bundle:
`labels/` and `samples.md` intentionally include answers for evaluation.
One JSONL line is one JSON record; the text field in each table record is
ready for BM25 or embedding. Database descriptions come from source domain
metadata, not generated answers. No INSERT values are put into the index.

Each label points to stable document IDs such as
`dbptd4c0ffee0002xtrain064571.agricultural_innovation_projects`. Source IDs,
split membership and reference queries are evaluator information, not retrieval
query text. The future online search receives the user's question.

## Verification and split policy

`rag/prepare_data.py` verifies the completed source manifest, SHA-256 hashes,
row counts, pinned revision, and disjoint source-context hashes. It parses
physical table references using SQL scopes: a CTE name is not counted as a
physical table. Unknown reference tables fail collection.

The Modal job uses **only mysql-eval-secret**. It queries information_schema
for the exact selected database names, checks table/column/key agreement, and
records actual MySQL types, nullability and comments. It executes no CREATE,
INSERT, DROP or reference-answer SQL. The source's previous successful query
execution is recorded separately; current metadata verification does not claim
to have rerun those queries or certified their meaning.

RL supplies retrieval-training tasks. Source-test contexts are deterministically
assigned to development or test by hash parity, keeping each schema on one
side. Evaluation questions are excluded if they match any SFT/RL question after
Unicode/case/whitespace normalization; repeated evaluation questions are also
excluded. On this source snapshot those overlap checks excluded no evaluation
questions. One RL query with no physical table was excluded.

This checks the post-training sources, **not the earlier pretraining .bin
files**, and does not detect paraphrases. These are useful retrieval labels,
not proof that the existing SQL generator has never seen an equivalent task.

## Limits of this first collection

- Most source databases contain one or a few tables. They are synthetic,
  independent examples, not a realistic shared warehouse.
- Many logical joins lack declared foreign keys. We index the 17 declared
  constraints, never infer edges from reference SQL. A missing edge does not
  mean a join is impossible. A larger relational benchmark can be added later
  to evaluate bridge-table discovery properly.
- Two databases can describe nearly identical domains. For **database-scoped
  retrieval**, supply the selected db_id and search within it. For **global
  routing**, search the entire catalogue using only the question, record
  ambiguity, and allow clarification. The source db_id is a reference label,
  not proof that only that database can answer an underspecified question.
- Including test-schema metadata in the catalogue is intentional: RAG searches
  known database schemas. Including test answers in the index would be leakage.
- The existing 2,048-token generator context still requires a prompt budget.
  The catalogue is not intended to be pasted into one prompt.

## Inspect or reproduce

Read `artifacts/rag/gretel-v1-20b6e311a001/samples.md` first, then one table
document and its matching label. Inspect the shared manifest:

```bash
uv run --locked modal volume get post-train-data \
  rag/gretel-v1-20b6e311a001/manifest.json - \
  --env main --profile sanika-05
```

Create a fresh, independently verified bundle using a new generated bundle ID:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 modal_rag_data.py
```

This spawns a CPU function and prints its Function Call ID and storage path.
The job writes a new directory, closes its files, then commits the Volume.
Consumers must verify manifest file hashes before using a downloaded copy.
The original posttrain_data directory is not changed.

For an offline source-only preview, download the four source files listed in
the ingestion manifest and run:

```bash
uv run --locked python -m rag.prepare_data \
  --source artifacts/rag/source --output artifacts/rag/my-new-preview
```

A preview records `live_verification.checked=false`; it must not be described
as currently verified against MySQL. Unit checks use no database credentials:

```bash
uv run --locked python -m unittest tests.test_rag_data
```

The next learning milestone is keyword retrieval over table documents and a
fixed development evaluation. This collection implements no retrieval ranking,
embeddings, generator changes or additional training.

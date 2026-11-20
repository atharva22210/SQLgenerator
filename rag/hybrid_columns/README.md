# Hybrid table retrieval, then column selection

This version starts from plain hybrid retrieval, independently of graph search
and cross-encoder reranking. Read `selection.py`, then `index.py`.

```text
question -> hybrid top-k tables -> score their columns -> add required keys
                                            -> reduced schema context
```

At build time we describe each column using its table name, column name/type,
optional comment and domain metadata. The existing frozen MiniLM encoder maps
these descriptions into normalized vectors `[total_columns,384]`. Snapshots
record their table/column order, encoder recipe and catalogue fingerprint.
Loading reuses the saved vectors; it does not re-encode columns.

At search time we encode the question and score columns in the selected tables
by cosine similarity. Default `--columns-per-table 3` selects three columns per
table, or all if the table has fewer. Then we preserve:

- All primary-key components.
- All local foreign-key components.
- Referenced columns in other selected tables, including non-primary unique
  keys. Table resolution stays within the same database.

The final column count can exceed three. For example, `customers.name` may win
semantic scoring while `customers.id`, `orders.customer_id` and the components
of a composite key survive even with low scores. Scores are ranking signals,
not confidence values; required columns bypass the semantic cutoff.

The selected table IDs and hybrid ranking do not change. Both `schema_text`
and `schema_sql` are reduced; returning the original SQL would defeat context
reduction. Original catalogue documents stay intact. The SQL is a quoted MySQL
schema sketch containing retained column types and declared primary/foreign
keys, not a migration: original defaults, checks, unique constraints and other
DDL details are not reconstructed. References to tables outside the selection
remain declared relationships, but those tables are not automatically added.

Output includes selected `columns`, keys, before/after counts and mandatory
column names. `--explain` additionally shows every candidate column's score and
whether it was selected semantically or preserved as a key. Debug explanations
are not the reduced generator context: use schema_text/schema_sql for that.

## Commands

Use the project's Python 3.13 environment and `uv sync --locked`. No new
dependency or model is introduced beyond hybrid's existing encoder. This runs
locally on CPU by default; no training or remote GPU is launched.

```bash
uv run --locked python -m rag.hybrid_columns.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-columns-v1/index.json build \
  --columns-per-table 3

uv run --locked python -m rag.hybrid_columns.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-columns-v1/index.json search \
  --query 'Which customers placed the largest orders?' --k 4 --explain

uv run --locked python -m rag.hybrid_columns.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-columns-v1/index.json evaluate \
  --split dev --k 4 --output artifacts/rag/hybrid-columns-v1/dev-global.json
```

`rag.cli` and `rag.service` also accept `--backend hybrid_columns`. Default k
is four for search/evaluation/API requests. Column selection never visits
tables excluded by hybrid's scope and server allowlist.

## What to evaluate next

Question: can we shrink schema context without losing columns needed by SQL?
Hypothesis: column scoring reduces context size while key retention preserves
join information. Baseline: plain hybrid. Variable: column selection only.
Keep catalogue, encoder, questions, k=4 and scope fixed. Record schema characters
and latency plus the existing table metrics; those table metrics alone cannot
detect an omitted filter or aggregation column. A fixed generator and SQL
execution evaluation (or column-level labels) are needed to test correctness.

Expected resources are local CPU, one-time column encoding, saved vectors and
one additional query encoding per search. The encoder's existing truncation
limit applies to each column description. Unit tests check key retention,
reduced text/DDL, source immutability, filtering, snapshot identity and shared
CLI/API behavior using fixed vectors. No dataset experiment has been run.

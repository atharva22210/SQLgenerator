# Hybrid + question-dependent foreign-key paths

This is the only graph variant. Plain BM25, dense and hybrid remain available.
Read `graph.py`, `connections.py`, then `index.py` and the shared CLI/evaluator. No graph
database or new dependency is required; this uses the existing frozen encoder,
NumPy, and Python's heapq for a readable weighted shortest-path algorithm.

```text
hybrid_graph/
  graph.py       # FK records, relationship text/vectors, costs and Dijkstra
  connections.py # Select connecting paths, prune cycles and bridge leaves
  index.py       # Hybrid seeds, graph expansion, budgets and snapshots
  cli.py         # python -m rag.hybrid_graph.cli
```

## Trace one request

1. Hybrid retrieves the top k table documents, called seeds (default k=4).
2. Table metadata supplies the declared foreign keys. Edges never come from
   reference SQL, question labels or guessed column-name matches. A target is
   resolved only inside the source database; unknown targets fail construction.
3. Each FK has a description such as `Table purchases columns (customer_id)
   reference table customers columns (id).` Embed these descriptions once when
   building the graph, using the same encoder as dense search. Keep individual
   FK column mappings even when two tables have several relationships.
4. Encode the current question. Compare it to normalized eligible edge vectors.
   Convert cosine similarity to s=(cosine+1)/2, clipped to [0,1]. Calculate:

   ```text
   edge cost = 1 - alpha * s
   ```

   Alpha defaults to 0.5 and must be strictly between zero and one. Every cost
   is positive, so Dijkstra can select the path with the smallest summed cost.
   A relevant edge with s=0.9 costs 0.55; an edge with s=0.2 costs 0.90.
5. For each same-database seed pair, find one minimum-cost path with no hop
   limit. Positive costs make Dijkstra terminate on the finite graph. FK edges can be traversed either way, while retaining their declared
   direction and columns as join evidence. For `customers <- purchases ->
   products`, traversal must be able to start at customers.
6. Consider paths by cost, breaking ties by table IDs then FK IDs. Select a
   path only if it joins separate groups of seeds and the whole path fits the
   additional-table budget. This is Kruskal-style selection over shortest paths.
7. Remove cycles using a minimum spanning forest on the selected FK edges.
   Repeatedly remove degree-one non-seed tables. Keep all seeds and only the
   bridge tables surviving cleanup. Shared bridges appear once in the context.

For example, if A-X-B costs 1.0, B-Y-C costs 1.2 and A-Z-C costs 1.8, the
first two paths already connect A, B and C. Skip A-Z-C, so Z never enters the
context. Four seeds need at most three selected paths; fewer can suffice if a
path passes through another seed.

For `customers -> purchases -> products` and `customers -> wishlists ->
products`, the question changes the edge costs. A purchase question may favor
purchases; a question about desired products may favor wishlists. This is a
heuristic based on the actual names in the schema, not guaranteed semantic
understanding. No relationship-specific business description is invented.

## Meaning of k, scores and limits

`--k` counts hybrid seeds, not final context tables, and defaults to 4 for graph
search, evaluation and HTTP requests. Other methods keep their default of 5.
There is no hop-limit parameter. Default `--max-added-tables 10` limits extra
schemas beyond those seeds. It is a table-count budget, not a tokenizer budget.
The snapshot records these build-time settings for every subsequent request.

`results` includes seeds then bridges; `seed_results` preserves original hybrid
hits. Seeds retain their RRF score, while bridges have `score: null` and
`retrieval_role: bridge`, because connectivity is not a hybrid relevance score.
`--explain` includes selected paths BEFORE cleanup, marked by `paths_stage`.
The `connections` list is the authoritative final pruned FK edge set, with
normalized similarities and individual costs. `connection_cost` sums those final
edges. `seed_groups` reports which seeds remain connected; `all_seeds_connected`
is true for one nonempty group. Diagnostics count redundant paths skipped and
list pruned table IDs. Normalized similarities
are not confidence probabilities. Graph diagnostics also count paths rejected
by the schema budget. `matched_documents` retains hybrid's pre-expansion hit
count; `context_table_count` counts the final schemas.

All traversal is restricted to permitted table IDs. Different databases are
never connected. Empty graphs, one seed, disconnected seeds or missing paths
leave the seed context intact. Cycles cannot improve a positive-cost path.
Equal-cost alternatives are resolved deterministically, not all included.

"Shortest" means minimum query-dependent cost, not necessarily fewest edges
or the correct SQL join. We find the best path for each seed pair, select
paths joining separate groups, then prune. This Steiner-tree heuristic is not
a globally minimum connected
subgraph, and no alternative path is substituted when the selected path fails
the remaining budget. A wrong initial seed can still add irrelevant context.
Undeclared relationships cannot be discovered by this implementation.

## Build, search and evaluate

Use the project's Python 3.13 and locked environment (`uv sync --locked`).
Like dense/hybrid, building may download MiniLM on first use. Everything here
defaults to local CPU; no remote GPU, training or SQL execution is launched.

```bash
uv run --locked python -m rag.hybrid_graph.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-graph-v2/index.json build \
  --graph-alpha 0.5 --max-added-tables 10

uv run --locked python -m rag.hybrid_graph.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-graph-v2/index.json search \
  --query 'Which customers purchased an iPhone?' --k 4 --explain

uv run --locked python -m rag.hybrid_graph.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/hybrid-graph-v2/index.json evaluate \
  --split dev --k 4 --output artifacts/rag/hybrid-graph-v2/dev-global.json
```

`rag.cli` and the shared local API also accept `--backend hybrid_graph`.
Snapshots include the nested hybrid index, graph settings, edge records and
edge vectors. Loading verifies the catalogue and encoder using existing checks,
rebuilds adjacency from the catalogue, compares it with saved edges, and reuses
table and edge embeddings without running the encoder on them again. Version-1
graph snapshots are rejected because path selection changed; build a new
version-2 snapshot. Standalone BM25, dense and hybrid snapshots are unchanged.

The common evaluator keeps recall/precision@k/complete/MRR on seeds, and adds
`context_recall`, `context_precision`, `context_complete`, `context_table_count`
and `context_char_count` for final context across all methods. Context precision
divides by the actual expanded result count, preventing precision above one
when more than k relevant tables are returned. Character count measures only
schema_text strings; it is not generator token count or the full prompt size.

## Correctness and the next experiment

Offline tests use fixed edge/query vectors to verify that two questions with
the same seed tables select different bridges. They also cover positive costs,
minimum-cost rather than minimum-hop selection, paths beyond four hops, ties, cycles,
scope isolation, whole-path budgets, duplicate bridges, snapshot reuse and
separate seed/context evaluation, redundant-path suppression and recursive leaf pruning. These tests do not prove MiniLM knows the
intended relationship semantics on real schemas.

Before a development experiment, predict whether graph expansion improves
complete coverage on multi-table questions and how much extra context it adds.
Baseline: plain hybrid with fixed catalogue, questions, encoder, k and scope.
Variable: graph expansion only, with alpha/budget settings fixed in advance.
Metrics: seed and context coverage/precision, schema count/characters, latency,
and the multi-table subset. Expected resources: local CPU, one-time edge
encoding and one additional question encoding plus graph traversal and connection pruning per
request. Avoid judging expanded coverage without its extra context cost; a
separate matched-context-budget comparison can test selection quality.

No dataset benchmark is run by this change. SQL execution accuracy and actual
prompt-token counts need a fixed SQL generator and execution evaluation stage,
which this retrieval-only milestone does not introduce.

For a later pruning comparison, keep k=4, edge costs, labels and scope fixed.
Compare the previous all-pairs union against connection selection plus pruning,
measuring context coverage and size together. No measured reduction in context
or SQL accuracy is claimed by correctness tests alone.

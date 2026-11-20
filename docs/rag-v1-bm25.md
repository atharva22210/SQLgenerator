# V1: explainable BM25 schema retrieval

Read `rag/bm25/index.py`, then `rag/common/cli.py`, then `rag/common/service.py`. Their docstrings
explain arguments, representations and the boundaries between retrieval and
evaluation. This milestone implements retrieval, not SQL generation or training.

## BM25 versus TF-IDF

Both use term frequency (TF) and inverse document frequency (IDF). We implement
BM25 directly, not a TF-IDF vectorizer followed by BM25. For each distinct query
term t appearing in document d, its contribution is:

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
contribution = IDF(t) * tf(t,d) * (k1 + 1)
               / (tf(t,d) + k1 * (1 - b + b * length(d)/average_length))
score = sum of matched-term contributions
```

N counts indexed tables; df counts tables containing a term; tf counts its
occurrences in one table document. Repetition gives diminishing returns. b
controls the penalty for longer documents. Defaults are k1=1.2 and b=0.75.
See [Elastic's BM25 explanation](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).

`customer_id` and `customerId` become customer/id; case is normalized. We do
not add stemming, synonyms or learned embeddings in this baseline. A question
using "buyers" can miss a table named "customers"; the dense implementation in
[rag/README.md](../rag/README.md) provides a comparison to evaluate separately.
Rare opaque database IDs and repeated document-template headings
are excluded from scoring. Search text uses table/domain/column/key metadata.

An inverted index maps terms to matching document IDs and their frequencies.
It avoids rescanning all schema strings per request. Global document statistics
are used even when filtering to one database, making that choice explicit.
BM25 scores are not probabilities or confidence values.

## Build and inspect

Generated indexes and reports belong in ignored artifacts/, not Git. No
dependency or model download is required. From the repository root:

```bash
uv run --locked python -m rag.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/bm25-v1/index.json build
```

Existing index paths are never overwritten; choose a new versioned path for
rebuilding. Search globally using only an English question:

```bash
uv run --locked python -m rag.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/bm25-v1/index.json search \
  --query 'What was the total amount of funds allocated for agricultural innovation projects in Rwanda in 2019?' \
  --k 5 --explain
```

The output includes schemas, document IDs, BM25 scores, timing and each term's
contribution. It does not include gold answers. `--db-id` optionally restricts
search to a database selected by the application. Unknown scopes fail; no
lexical match returns an empty list, never arbitrary zero-score tables.

## Development evaluation

Question: how well does the fixed lexical baseline recover required tables?
Hypothesis: global retrieval is harder than retrieval within a known database
because many synthetic databases have overlapping terminology. Fix k=5,
k1=1.2, b=0.75; change only the candidate scope. Compare table recall,
precision@k, all-required-table coverage and first-relevant reciprocal rank.
Resource use is local CPU, with no generation calls, GPU or database queries.

```bash
uv run --locked python -m rag.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/bm25-v1/index.json evaluate \
  --split dev --k 5 --output artifacts/rag/bm25-v1/dev-global.json

uv run --locked python -m rag.cli \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/bm25-v1/index.json evaluate \
  --split dev --k 5 --scoped --output artifacts/rag/bm25-v1/dev-scoped.json
```

Gold IDs are compared only after searching. Global mode does not receive
db_id. Scoped mode supplies db_id, so its score must not be reported as database
routing accuracy. Both reports include all questions and a multi-table subset.
No test questions are evaluated in this milestone; reserve them until settings
are fixed. The test split is available via an explicit `--split test`.

For gold={customers,orders} and retrieved={customers,employees,suppliers},
recall=1/2 and complete coverage=0. Precision@k divides by REQUESTED k, even if
the result list is shorter. Thus database-scoped precision@5 can be low when a
database has only one table; interpret it with recall, coverage and candidate
size. Complete coverage is more useful than first-hit MRR for multi-table SQL.

The JSON report preserves per-question failures, index/source identity, label
file hash and configuration. p50/p95 measure in-process retrieval only. They
exclude startup, API/network, SQL generation and execution. No SQL accuracy or
end-to-end latency is measured yet. These small synthetic databases are not a
representative enterprise benchmark; alternative equivalent table choices and
underspecified database routing can also make source labels imperfect.

## Local API and operational boundaries

Optional: set RAG_API_TOKEN through a LOCAL hidden prompt (zsh on your Mac),
using a random secret of at least 32 characters. Do not paste it into chat:

```bash
read -s 'RAG_API_TOKEN?Enter the local API token: '
export RAG_API_TOKEN
uv run --locked python -m rag.service \
  --bundle artifacts/rag/gretel-v1-20b6e311a001 \
  --index artifacts/rag/bm25-v1/index.json \
  --allow-db dbptd4c0ffee0002xtrain064571
```

Another terminal can check `http://127.0.0.1:8080/health`. POST /search accepts
JSON with query, k, optional db_id and optional explain. Supply the bearer
token in the Authorization header through your local API client. `--allow-db`
is server-controlled and repeatable; request bodies cannot expand it.

The service implements a single trusted application identity, NOT per-user
authentication. It binds loopback only and handles requests serially. Before
external deployment, add TLS/identity/rate limiting via a proxy and choose a
managed serving stack with measured concurrency limits. A secret alone is not
multi-tenant authorization; database execution must enforce its own permissions.

Implemented safeguards:

- Verify completed catalogue checksum/count and reject stale saved indexes.
- Version analyzer/config/source identity; publish JSON snapshots atomically
  without overwriting; deserialize no pickle.
- Apply database permissions before top-k; stable tie-breaking.
- Limit questions to 2,000 characters, k to 20, API bodies to 16 KiB and socket
  reads to five seconds. Return no-match explicitly; similarity is not confidence.
- Log request ID, status and elapsed time, omitting question text and credentials.
- Keep labels/ and reference SQL out of indexing, and avoid query-result caches
  that could mix authorization scopes. There is no schema auto-refresh: rebuild
  an index when the catalogue changes; document its snapshot identity.

The search API returns schema context only. It executes no generated SQL.

## First observed development results

On 250 development questions (50 requiring multiple tables), with the fixed
defaults above, the first run measured:

| Retrieval scope | Mean recall@5 | Complete coverage@5 | Multi-table complete coverage@5 |
| --- | ---: | ---: | ---: |
| Global, question only | 66.73% | 61.6% (154/250) | 12% (6/50) |
| Correct database supplied | 99.13% | 98.8% (247/250) | 96% (48/50) |

Local in-process p95 retrieval was approximately 1.39 ms global and 0.11 ms
scoped, excluding startup/network/generation. This is one small local run,
not a concurrent service benchmark or an end-to-end latency guarantee.

The observed gap supports the hypothesis that global selection is harder in
this corpus. The scoped result is especially easy because most databases have
very few candidate tables; it is not evidence of enterprise-scale success.
Global hits can span multiple databases: downstream code must route/select a
database or clarify ambiguity before constructing a generator prompt.
The next experiment should investigate database routing and missing join
tables. Inspect per-question failures before choosing a change. We have not
tuned settings against the held-out test set or measured SQL execution accuracy.

## Checks

```bash
uv run --locked python -m unittest tests.test_rag_bm25
uv run --locked python -m unittest discover -s tests
```

Tests cover a hand-calculated score, ranking explanations, identifier handling,
no-match responses, permission filtering before top-k, invalid requests,
stale snapshots, no-clobber writes, evaluation denominators and authenticated
HTTP behavior. Socket tests need loopback access.

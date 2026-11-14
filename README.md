# SQLgenerator

A learning project for training an autoregressive text-to-MySQL model from
scratch, with an emphasis on pre-training and distributed execution. The
tokenizer, dataset, and prompt format are still to be chosen.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and run from its root:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests
```

`.python-version` selects Python 3.13.15. `pyproject.toml` declares the project
dependencies; `uv.lock` records the resolved direct and transitive versions.
Both teammates use the same Git revision and `uv sync --locked` to build their
own ignored `.venv`. No activation is needed with `uv run`. uv itself is a
bootstrap tool; use uv 0.12.3 or later (the Modal image uses 0.13.7).

Use `uv add <package>` for intentional dependency changes, review changes to
both the configuration and lockfile, and commit them together. Avoid manually
installing packages into `.venv`: the next sync removes undeclared packages.

PyTorch is pinned to 2.10.0. macOS uses its native PyPI build; Linux and Windows
use PyTorch's CUDA 12.8 index. The Modal image uses the same lockfile and Python
version. This reproduces the Python dependencies for each platform, not the
host GPU driver or identical floating-point results across hardware.
PyTorch is in the default `model` dependency group. The CPU-only tracking
image omits that group, avoiding CUDA libraries while keeping W&B's locked
dependencies. Local setup and GPU images include the group by default.

## Modal and W&B setup

Each teammate authenticates with their own account:

```bash
uv run --locked modal setup
uv run --locked modal profile list
```

The existing project workspace is `sanika-05`, with environment `main`, Volume
`sqlgen-data` (v2), and Secret `wandb-secret`. Teammates need access to the same
workspace to share these resources; do not share personal Modal tokens.
The volume is for datasets and checkpoints. GPU checks do not write to it;
the separate volume check below writes a small marker under `volume_checks/`.

For a separate workspace, create its volume once:

```bash
uv run --locked modal volume create sqlgen-data --version 2 --env main
```

In the Modal dashboard, create `wandb-secret` with the field `WANDB_API_KEY`.
Enter credentials directly there, never in Git, chat, or command arguments.
W&B team/entity: `dkbsingh7-iit-delhi`; smoke-test project: `sqlgen-infra`.
The logging preflight creates that project privately if it does not exist;
the API key must have permission to create projects and log runs in the team.
Existing project visibility is preserved.

To verify W&B authentication and logging without allocating a GPU:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus none --log-wandb --wandb-entity dkbsingh7-iit-delhi
```

This creates a tracking-only run with `tracking_connected=true` and
`gpu_checks_run=0`; it makes no claim that GPU checks passed.

## MySQL sandbox and eval checks

The local and remote environments use the locked PyMySQL, cryptography,
DBUtils and python-dotenv dependencies. Run `uv sync --locked` after updating.
Copy `.env.example` to `.env` and fill in your six MySQL fields locally.
`.env` and `.env.*` are ignored; the empty `.env.example` is the only exception.
Existing environment variables (including Modal Secrets) take precedence over
the optional repo-root `.env`, regardless of your current working directory.

Download your service's CA certificate from Aiven and set
`MYSQL_SSL_CA=~/Downloads/ca.pem` in `.env`, or pass `--ca` before a subcommand.
TLS requires a valid certificate and matching hostname, with TLS 1.2 or later.
There is no insecure TLS fallback. `MYSQL_SSL_CA_PEM` accepts certificate text
and takes precedence over the local path, so the same client works in Modal.
If neither CA setting is supplied, the system trust store is used.

From the repository root:

```bash
uv run --locked python -m infra.verify_mysql_setup init-admin
uv run --locked python -m infra.verify_mysql_setup test-read
uv run --locked python -m infra.verify_mysql_setup test-modal-gpu --profile sanika-05 --env main
```

`init-admin` creates `sandbox_db`, `team_check`, an initial row, and an empty
`__sqlgen_drop_probe` table without replacing existing rows. It creates
`MYSQL_EVAL_USER` at host `%` if absent, requires SSL, and grants only `SELECT`
and `SHOW VIEW` on this database. MySQL users are accounts, separate from the
rows in `team_check`. Database-name wildcard handling respects MySQL's
`partial_revokes` setting so `sandbox_db` does not match other databases.
Existing eval passwords are preserved; `.env` must match the account's password.
Broader existing grants or roles cause setup to stop for manual review; setup
never revokes permissions belonging to another workflow. Each statement commits
independently, and MySQL DDL implicitly commits, so setup is rerunnable but not
atomic. Use a dedicated eval account, distinct from admin.

`CREATE USER` and `GRANT` apply without a flush. If explicitly needed, use
`init-admin --flush-privileges`; managed services may restrict that operation.
See [MySQL's FLUSH documentation](https://dev.mysql.com/doc/refman/8.0/en/flush.html).

`test-read` connects only as eval, verifies encryption and narrow grants,
prints rows from `sandbox_db.team_check`, and attempts to drop the empty probe.
Only MySQL error **1142** counts as a successful denial. Other errors fail the
check. An unexpectedly successful DROP fails the check and removes only the
disposable probe; rerun `init-admin` after correcting permissions. A SELECT and
denied DROP are an infrastructure check, not a complete SQL evaluation sandbox.

For the GPU command, create a Modal Secret named **`mysql-eval-secret`** in
workspace `sanika-05`, environment `main`, using the provider dashboard or a
local credential prompt. Include only `MYSQL_HOST`, `MYSQL_PORT`,
`MYSQL_EVAL_USER`, `MYSQL_EVAL_PASS`, and `MYSQL_SSL_CA_PEM` (the full contents
of your CA certificate). The existing `mysql-secret` contains admin credentials
and is intentionally not used for eval GPU jobs. Never upload `.env` or the
admin password to an eval image. `--secret-name` selects another eval-only
Secret; `--profile` selects your local alias for the shared workspace.

The Modal check runs the same `test_read()` on one L4, falling back to one A10.
It checks the GPU with `nvidia-smi`; no PyTorch/CUDA Python packages, training,
W&B or Volume are needed. The image installs the locked project dependencies
without the model group and includes only the `infra` Python package. Each
remote invocation has a 120-second execution timeout; the caller waits at most
240 seconds after submission and cancels an unfinished call. Image build and
startup have separate lifetimes. The result includes the GPU, TLS cipher,
rows and permission error code. This verifies database access from that GPU
container, not multi-GPU communication or training throughput.

Reuse the client in Python:

```python
import pymysql
from infra.db_client import execute_query

try:
    rows = execute_query("SELECT * FROM sandbox_db.team_check WHERE id = %s", (1,))
except pymysql.Error as exc:
    mysql_error_code = exc.args[0]
```

Eval is the default; setup can explicitly pass `mode="admin"`. Each process
uses a separate lazy pool per mode, with at most four dedicated connections;
exhaustion raises instead of waiting indefinitely. Queries commit on success,
roll back on error, and return connections to the pool. `execute_query` accepts
one statement, returns dictionaries (`[]` for statements with no result), and
preserves original PyMySQL exceptions. `connection(mode)` supports multiple
statements in one transaction; `close_pools()` releases idle connections. Create
pools after forking, and close pools before changing connection configuration.

## Day 4: verified post-training data

`data/ingest_mysql_benchmarks.py` streams
[`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
at pinned revision `740ab236e64503fba51be1101df7a1be83bf455d`. The source has
100,000 train rows and 5,851 test rows. It executes each selected row's original
`sql_context` using admin credentials, then its original answer `sql` using
the existing eval-only account in a read-only transaction. It does not
transpile SQL, add primary keys, fix queries, or validate the meaning of answers.
A successful empty result is valid execution evidence.

The approved shared server reports **MySQL 8.4.11**, rather than the curriculum's
8.0. Outputs record the actual version and SQL mode; they do not certify 8.0
compatibility. `ONLY_FULL_GROUP_BY` must remain enabled. Aiven requires primary
keys by default; ingestion uses its documented
[`SET SESSION sql_require_primary_key = 0`](https://aiven.io/docs/products/mysql/howto/create-tables-without-primary-keys)
for admin connections to create these small synthetic tables. The service's
global setting stays unchanged.

The new shared v2 Volume is **`post-train-data`**, mounted at `/vol`:

| Path inside the container | Verified rows | Source | Live MySQL database |
| --- | ---: | --- | --- |
| `/vol/posttrain_data/sft_train_45k.parquet` | 45,000 | train | No; temporary schemas are reused and removed |
| `/vol/posttrain_data/rl_train_2000.parquet` | 2,000 | train, after SFT selection | Yes |
| `/vol/posttrain_data/eval_test_500.parquet` | 500 | test | Yes |

Test rows are selected first so their exact context hashes can be excluded
from both training outputs. SFT and RL follow source order; RL also excludes
contexts used by SFT. Exact `(prompt, context, SQL)` duplicates are excluded.
This prevents identical-context leakage across splits, not every possible
semantic duplicate. No shuffling or SQL repair is applied. Unsupported
non-SELECT answers, unsafe contexts, parser failures, and MySQL SQL/data errors
are logged in `validation.jsonl`. Unsupported stored-function calls also fail
validation (error 1370); eval is never granted EXECUTE. Authentication,
database/table permission, network, quota, and storage
failures stop ingestion. If a source is exhausted before the target count,
the job fails rather than publishing undersized datasets.

Each Parquet row retains all source columns and adds `source_split`,
`source_index`, `dataset_revision`, `context_sha256`, `db_name`, and
`result_row_count`. SFT `db_name` is null. Live names use the run's isolated
namespace, such as `dbpt012345abcdefxtrain000001`; consumers must read `db_name`
rather than assume consecutive names. The eval account receives only SELECT
and SHOW VIEW on each exact database, with no database-name wildcard grant.
The existing permission checks recognize these exact ingestion names.

Run `uv sync --locked` first. The remote job uses the same Python 3.13.15 and
locked dependencies without the model group. It receives the existing
`mysql-secret` and `mysql-eval-secret`; the latter must include CA PEM text
as described above. No `.env`, personal token, GPU, or W&B run is uploaded or
allocated. Four workers use up to nine MySQL connections plus a short-lived
preflight connection; the smoke check below uses two workers. CPU, storage,
and MySQL service usage still apply. Containers use `ap-southeast` to stay near
the Singapore database endpoint; Modal applies its region-selection pricing.
Full runtime depends on database latency
and rejection rate; it is not bounded by the curriculum's 75-minute estimate.

First run a small end-to-end check in a separate Volume directory. Reuse its
run ID to resume it; choose a fresh 12-character lowercase hex ID for a new check:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 modal_posttrain.py --run-id d4c0ffee0001 --smoke --sft-rows 20 --rl-rows 2 --test-rows 2 --workers 2
```

Its validation checkpoints go to `/vol/checks/d4c0ffee0001/`. To publish and
independently verify those smoke Parquet files after reviewing their counts:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_posttrain.py --run-id d4c0ffee0001 --smoke --sft-rows 20 --rl-rows 2 --test-rows 2 --workers 2 --publish-only
```

Smoke RL/test databases remain available for checks. The full run uses
`/vol/posttrain_data/`:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 modal_posttrain.py --run-id d4c0ffee0002 --submit-only
```

This command validates only. Once all targets are reached, it writes the actual
filtered counts to `ready.json` and leaves the final Parquet files unwritten.
`--submit-only` prints the Function Call ID and exits the local launcher while
the detached remote job runs. Inspect `progress.json` or the App dashboard for
progress; `ready.json` is written only after validation finishes successfully.
Report/review those counts before publishing. Publication uses only the accepted
rows in the validation ledger; rejected rows and checkpoints are not included
in any Parquet dataset:

```bash
uv run --locked modal volume get post-train-data posttrain_data/ready.json - --env main --profile sanika-05
uv run --locked modal run --detach --env main --profile sanika-05 modal_posttrain.py --run-id d4c0ffee0002 --publish-only
```

Keep the run ID. The same command resumes from the last complete ledger entry
after failure; the output's config must match. Run-scoped pending databases
are checkpointed before creation so recovery can remove only unfinished
databases owned by that run. A server-side ingestion lock prevents simultaneous
runs from modifying the data. Existing unrelated databases are preserved.
The ledger and `progress.json` are committed every batch (up to 1,000 SFT
candidates or 100 RL/test candidates). A failure can require replaying the
unfinished batch. Final Parquet files
are published only after reaching every requested count, with hashes and the
server details in `manifest.json`.

After ingestion, an independent container mounts the Volume read-only, checks
all file hashes, row counts, and split disjointness, and reruns all 2,500 live
queries with eval-only credentials. Repeat it without ingestion:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_posttrain.py --verify-only
uv run --locked modal volume get post-train-data posttrain_data/progress.json - --env main --profile sanika-05
uv run --locked modal volume get post-train-data posttrain_data/manifest.json - --env main --profile sanika-05
```

`manifest.json` establishes that ingestion finished; inspect the independent
verification result to confirm the retained databases are still accessible.
To verify the smoke output, add `--smoke --run-id d4c0ffee0001` to the
`--verify-only` command. For a local run with the repo's ignored `.env`, use
`uv run --locked python -m data.ingest_mysql_benchmarks --output artifacts/posttrain_data --run-id 012345abcdef`.
Add `--validate-only` to that local command to separate validation from writing
Parquet files. Local output is not automatically uploaded to Modal.

## Shared-volume check

Both teammates must access workspace `sanika-05`, environment `main`, and the
existing `sqlgen-data` Volume. Invite your teammate to that workspace using
Modal's dashboard; they authenticate with their own account and select that
workspace in `modal setup`. A similarly named Volume in their own workspace
is separate storage. Profile names are local aliases: use whichever profile
points to `sanika-05` if it differs from the name in these examples.

Run the CPU-only check once to create a uniquely named marker:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_volume_smoke.py
```

The writer exclusively creates `volume_checks/<marker>.json`, closes it, and
calls `volume.commit()`. A separate reader container mounts the same Volume
read-only, calls `reload()`, and checks the file's SHA-256 hash. The result
must have matching Volume IDs and checksums, and different writer and reader
container IDs. Existing files are never overwritten; the marker remains for
your teammate to verify later. This check uses no GPUs or W&B credentials.

Share the read-only command printed by the script with your teammate. They
run it with the same code and their own authenticated profile; it supplies
`--marker`, `--expected-sha256`, and `--expected-volume-id` and performs no
write. It checks the Volume ID as well as the file contents. This step verifies
access from their account, beyond the cross-container test on your account.
The same read-only command can be rerun in a later Modal App to verify that
the file outlives the original App.

Before sharing the new script, your teammate can also read the marker using
their installed Modal CLI, with their virtual environment activated. Replace
`MARKER` with the UUID printed by the check:

```bash
modal volume get sqlgen-data volume_checks/MARKER.json - --env main --profile sanika-05
```

The `-` prints the stored JSON to the terminal without creating a local file.

This is shared cloud storage mounted inside Modal containers, not automatic
synchronization of laptop directories. Use `modal volume get` / `put` for
explicit file transfers between a laptop and the Volume. Writers should
commit completed changes and readers should reload before checking updates;
avoid two teammates overwriting the same checkpoint path concurrently.

## GPU smoke tests

These commands run billable remote jobs. Each GPU function has a 300-second
timeout, its worker subprocess has a 180-second timeout, and NCCL has a
60-second collective timeout. Startup has separate deadlines. The connected
local runner requests cancellation after three minutes for CPU checks or six
minutes for GPU checks, including queue time; it depends on the local client
remaining alive and connected. Containers scale down after a short idle window.
No training or persistent deployment is started.
`--gpu-wait-seconds 600` allows a longer allocation wait without increasing
the GPU function's execution timeout. A wait timeout means the call did not
finish in that window; it is not evidence of an all-reduce failure if no
worker reached the test.

For a two-L4 check that survives a laptop/client disconnect, run the remote
CPU controller as a detached entrypoint:

```bash
uv run --locked modal run --detach --env main --profile sanika-05 modal_smoke.py::verify_dual_l4 --wandb-entity dkbsingh7-iit-delhi --gpu-wait-seconds 600
```

The controller checks W&B access, submits one two-L4 call, and enforces its
waiting deadline inside Modal using an asynchronous wall-clock timeout. It
requests child cancellation on timeout or interruption. Successful GPU results
are printed before W&B logging and returned as a JSON report with the run URL.
The CPU controller has a 20-minute execution limit covering setup, GPU waiting,
logging, and cleanup; the GPU function still has a five-minute execution limit.
If cancellation cannot be confirmed, it logs that explicitly: inspect and stop
the App rather than assuming the child is gone. No automatic retry is scheduled.
Use the App link from the CLI to view scheduler messages, calls, and logs; stop
the finished detached App after collecting its results.

Run the single-GPU check first:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus 1
```

Then run the communication check:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus 2
```

To run both and record one W&B run:

```bash
uv run --locked modal run --env main --profile sanika-05 modal_smoke.py --gpus both --log-wandb --wandb-entity dkbsingh7-iit-delhi --wandb-project sqlgen-infra
```

`modal_smoke.py` allocates the GPUs, then launches `scripts/gpu_smoke_worker.py`
with `torchrun`, one process per GPU. A **rank** identifies a process in the
communication group; **world size** is the number of processes; **local rank**
selects its GPU on this machine. For two L4s, world size is 2 and local ranks
are 0 and 1. NCCL is the communication backend used by these NVIDIA GPUs.

Each worker prints the visible GPU count, its assigned device and GPU name,
and asserts a small CUDA computation. For two GPUs, rank 0 starts with `[1]`
and rank 1 with `[2]`; a SUM all-reduce must leave **both** with `[3]`. Both
workers assert the result and destroy the process group on completion.
The single-GPU check leaves its tensor at `[1]` without initializing NCCL.

When logging is requested, a CPU preflight checks project access before GPU
allocation. After the GPU checks, a separate CPU function uses `wandb-secret`
to log pass indicators, elapsed time, and a table of worker results. GPU
workers do not receive the W&B credential. Elapsed time excludes allocation
waiting and includes worker interpreter startup; it is not a communication
benchmark. Passing these tests verifies basic GPU
execution and communication, not training correctness or scaling efficiency.

If a check fails, inspect the first worker error in Modal's logs. CUDA checks
fail explicitly if the allocation is missing or unexpected; torchrun worker
failures propagate to the caller. A W&B authentication or permission failure
is reported as an error rather than silently switching to offline logging.

## Model

```python
import torch
from sqlgenerator import CausalLanguageModel, ModelConfig

model = CausalLanguageModel(ModelConfig(vocab_size=tokenizer.vocab_size))
input_ids = torch.tensor([[...]])
logits, loss = model(input_ids, labels=input_ids)
```

Provide token IDs from the tokenizer and data pipeline selected for the target
database schema. `generate` returns token IDs, leaving decoding and SQL
validation to the application.

## Parallelism

For one process, use the model directly. For a single-process multi-GPU
prototype, call `wrap_model(model, torch.device("cuda"), "data_parallel")`.
For scalable training, launch one process per GPU:

```bash
torchrun --nproc_per_node=2 <your_training_entrypoint>.py
```

Inside each worker, call `initialize_distributed()` and pass its returned
device to `wrap_model(model, device, "ddp")`. Dataset sharding, optimizer
configuration, checkpoint policy, and training loop are intentionally left to
the training application.

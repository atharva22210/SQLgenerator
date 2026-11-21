# Memory and sharding on two A10s

[fsdp_checkpointing.py](fsdp_checkpointing.py) defines the benchmark cases, wraps
model blocks for sharding and checkpointing, and exports GPU profiles.

C3 has 24 blocks at width 1536, with 24 query heads, 12 KV heads, and SwiGLU width 4096. It has 622,929,408 parameters before the tied vocabulary matrix. Add `1536 × vocab_size` for the loaded tokenizer and multiply by 16 bytes to estimate DDP model-state memory on each GPU. The earlier ~26 GB estimate for state plus activations used a hypothetical activation budget. With the saved 151,670-token vocabulary, C3 has 855,894,528 trainable parameters and about 12.75 GiB of DDP model state per GPU. The baseline tests the OOM prediction.

The suite runs plain DDP, DDP with block activation checkpointing, FSDP `SHARD_GRAD_OP`, FSDP `FULL_SHARD`, and `FULL_SHARD` with block activation checkpointing. Both checkpointed cases replay the first 12 of 24 blocks by default. All cases share the seed, batches, BF16 compute, FP32 optimizer state, and checkpointed vocabulary-loss chunks. By default, each rank accumulates two microbatches of four rows of 2,048 slots for each of 20 updates. DDP synchronizes the accumulated gradients on the last microbatch. FSDP reduce-scatters gradients on every microbatch so accumulated gradients remain sharded. FSDP `no_sync` would defer that reduction while holding full gradients and would still need parameter gathers.

Start with a three-update, short-context check of both GPUs, including the same full C3 parameter shapes. Supply the team's W&B entity.

```sh
uv run --locked modal run --detach --env main --profile sanika-05 -m distributed_training.launch \
  --part c --max-steps 3 --micro-batch-size 1 --seq-len 64 \
  --val-batches 2 --wandb-entity YOUR_ENTITY
```

For the full experiment, remove the four smoke overrides. Part C uses the latest completed Grouped SQL build by default. Pass `--grouped-sql-dir /vol/pretrain_data/grouped_sql/<corpus_id>` to pin a specific build. The suite holds two A10s while it runs each case in sequence, up to 100 total optimizer updates. Record your predictions about which cases fit and which expose more communication before submitting.

W&B groups launches by experiment topic, Grouped SQL build ID prefix, sequence length, and launch time. The training run names spell out the distributed method and the number of checkpointed blocks. Multi-case launches also have a comparison run. The comparison has training and fixed-sample filtered SQL validation curves, a memory and throughput table, and per-GPU bars for time in NCCL kernels, other GPU kernels, and their overlap. Overlapping kernels in a category count once. The bars use kernel names to classify activity and omit memory copies. Open either rank's Chrome trace from the W&B artifact in Perfetto to inspect where GPU work waits. The final update supplies the trace. Steady throughput excludes that update and warmup, and the comparison table takes peak allocated memory from unprofiled training updates. Validation runs every five updates and at the end by default. Both FSDP ranks evaluate the same fixed sample collectively, with rank 0 logging its result.

CUDA OOMs appear in the comparison table with rank and phase details in each case's `result.json`. The suite proceeds to the next case after a recorded CUDA OOM. Results and both rank traces are saved under `/vol/checkpoints/distributed_training/<suite>`. Part C saves metrics and traces for these short benchmarks. DDP correctness and synchronization runs also save final model snapshots.

To rerun just the successful configuration, select `--part c --case C3-fix4-full-shard-checkpoint` and pass the completed grouped SQL directory with `--grouped-sql-dir`.

In the training run, open `charts/training_loss` and `charts/sql_validation_loss` for separate loss plots by optimizer update. The validation chart has one point per evaluation update. The native `train/grad_norm` series is the global gradient norm before clipping. `charts/gpu_profile` shows NCCL time, other GPU kernel time, and overlap for each GPU from the final profiled update. For a single-case launch, all three panels and the trace artifact are on the training run.

## Recorded C3 result

The historical full-context run used the earlier corpus, batch 4 per GPU, and
2,048-token rows. Plain DDP, DDP with 12 blocks checkpointed, ZeRO-2-style
sharding, and full sharding without checkpointing ran out of memory. Full
sharding with 12 blocks checkpointed finished 20 updates at 18.69 GiB peak
allocated per GPU and 4,676 non-padding tokens/sec across both A10s.
[Full-context comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/0ii241bf)

All five methods fit in the short-context smoke test. Those traces support
comparing communication behavior when every method fits. The full-context run
answers which methods fit at the larger workload. Results on the Grouped SQL
build remain to be measured.
[Short-context comparison](https://forge.coreweave.com/wandb/dkbsingh7-iit-delhi/sqlgen-day6/runs/s37qyy4b)

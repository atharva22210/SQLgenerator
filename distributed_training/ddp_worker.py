"""Torchrun worker for DDP correctness, communication, and sharding experiments.

Launched by distributed_training.launch, not directly by `modal run`. Paths and
hyperparameters arrive in an explicit JSON file shared by all local ranks.

Trace one update here:
    deterministic CPU batches -> zero gradients -> local backwards with DDP
    -> clip the synchronized global gradient -> AdamW step -> global metrics.

DDP is data parallelism: each rank stores a FULL model, gradients and optimizer
state. Input data is assigned explicitly; DDP does not shard the dataset for
us. CUDA/NCCL communicates gradients; only rank 0 owns W&B and output files.
The initial and validation scans are outside the measured training-update time.
"""
from dataclasses import asdict
from datetime import timedelta
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloader import Corpus, make_batch
from distributed_training.fsdp_checkpointing import export_profile, prepare_model, profile_update
from distributed_training.ddp_core import accumulated_backward, batch_rng, microbatch_ids
from model.transformer_35m import Transformer, TransformerConfig
from data.grouped_corpus import resolve_grouped_sql
from training.config import DATA_DIR, TrainConfig, learning_rate, volume
from distributed_training.sql_validation import load_training_and_sql_validation, evaluate_sql


def run(settings: dict) -> None:
    """Run one labeled A/B experiment, using the same code path for all cases.

    Args:
        settings: JSON-compatible dictionary written by the Modal parent.
            train: TrainConfig fields (batch size, accumulation, LR, etc.).
            model: TransformerConfig overrides, such as width and layers.
            world_size: Expected number of GPU workers, checked against torchrun.
            duplicate: Select repeated microbatch IDs for the intentional A3 bug.
            no_sync: Suppress gradient sync on all but the last microbatch.
            benchmark_warmup: Initial updates excluded from throughput summary;
                learning-rate warmup is configured separately in train.
            output_dir: New directory for this case's final snapshot/results.
            run_name/group: Individual W&B label and shared suite identity.
            wandb_entity/wandb_project: Experiment-tracking destination.

    Returns:
        None. Rank 0 writes result.json/final.pt and logs scalar metrics.

    settings contains train/model dataclass fields plus world_size, duplicate,
    no_sync, run_name, output_dir, benchmark_warmup, wandb_entity and project.
    Results include global useful-token throughput, per-rank peak memory,
    filtered validation, and a final checkpoint. No implicit checkpoint resume:
    every experiment starts from the same seed and a new output directory.
    """
    # torchrun launches the SAME Python module in each process and injects:
    # RANK = identity among all workers; LOCAL_RANK = GPU index on this host;
    # WORLD_SIZE = total cooperating workers. They coincide on our single host,
    # but are different concepts when training spans several machines.
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != settings['world_size']:
        raise ValueError('torchrun world size differs from experiment settings')
    # Bind before constructing CUDA tensors or NCCL collectives. Otherwise
    # two workers might both allocate on GPU 0 and leave GPU 1 unused.
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('These experiments require CUDA BF16 support')
    distributed = world > 1
    if distributed:
        # A process group is the set of participants in collective operations.
        # NCCL is the CUDA communication backend. The long startup timeout also
        # covers corpus indexing/filtering while another rank waits for rank 0.
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    run_handle = None
    part_c = settings.get('part_c', False)
    method = settings.get('method', 'ddp')
    sharded = method != 'ddp'
    parameter_count = None
    update = None
    phase = 'startup'
    output = Path(settings['output_dir'])
    try:
        config = TrainConfig(**settings['train'])
        grouped_sql = settings.get('grouped_sql_dir')
        grouped_sql_dir = Path(grouped_sql) if grouped_sql else None
        if rank == 0:
            # Never silently resume/overwrite a previous experiment. The
            # launcher supplies a fresh UUID-based suite directory each time.
            output.mkdir(parents=True, exist_ok=False)
            volume.reload()
            grouped_sql_dir, _ = resolve_grouped_sql(DATA_DIR, grouped_sql_dir)
            tokenizer, corpus, validation = load_training_and_sql_validation(grouped_sql_dir, reload_volume=False)
            identity = {'vocab_size': len(tokenizer), 'pad_id': tokenizer.pad_token_id,
                        'eos_id': tokenizer.eos_token_id, 'grouped_sql_dir': str(grouped_sql_dir)}
        else:
            identity = None
        if distributed:
            # Share only the small tokenizer identity, not the large corpus.
            # broadcast_object_list mutates the receiver's list in place.
            # Both ranks must use the same PAD/EOS IDs and vocabulary size.
            objects = [identity]
            dist.broadcast_object_list(objects, src=0, device=device)
            identity = objects[0]
        grouped_sql_dir = Path(identity["grouped_sql_dir"])
        settings = {**settings, "grouped_sql_dir": str(grouped_sql_dir)}
        if rank != 0:
            if sharded:
                # Rank 0 has finished reloading the shared mount before the
                # broadcast. A second reload could disrupt its open memmaps.
                _, corpus, validation = load_training_and_sql_validation(grouped_sql_dir, reload_volume=False)
            else:
                corpus = Corpus(DATA_DIR, 'train', identity['eos_id'], grouped_sql_dir=grouped_sql_dir)
        # Identical initialization is essential for a fair A1/A2 comparison.
        # Do NOT add rank here. Data randomness is separately keyed by each
        # global microbatch ID, rather than perturbing model initialization.
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        # TrainConfig is the source of initialization settings for every rank;
        # model overrides below describe architecture, not a second init choice.
        model_settings = {**settings['model'], 'initializer_range': config.initializer_range}
        raw_model = Transformer(TransformerConfig(vocab_size=identity['vocab_size'],
                                                 max_seq_len=config.seq_len,
                                                 **model_settings))
        parameter_count = sum(p.numel() for p in raw_model.parameters())
        # Classify by the original shape before FSDP replaces it with a shard.
        decay = [p for p in raw_model.parameters() if p.ndim >= 2]
        no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
        prepared = prepare_model(raw_model, method, settings.get('checkpoint_blocks', 0), device)
        # Keep raw_model for rank-only validation and prefix-free state_dicts;
        # use the DDP wrapper for EVERY distributed training forward so its
        # autograd hooks can schedule gradient reductions during backward.
        # broadcast_buffers=False is appropriate for this model: RoPE tables
        # are deterministic and it has no changing BatchNorm running statistics.
        model = (DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False,
                     gradient_as_bucket_view=True) if distributed and not sharded else prepared)
        # BF16 autocast changes eligible compute; stored weights/gradients and
        # Adam moments stay FP32. Bucket views avoid a separate gradient copy.
        # Decay matrix-shaped weights; leave 1-D normalization scales alone.
        # Tied embeddings occur once in parameters(), avoiding duplicate updates.
        # fused=True selects fused optimizer kernels, not mixed precision itself.
        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': config.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=config.peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        # Separate logging runs on every rank would duplicate metrics and
        # create competing histories. One rank logs already-combined results.
        if rank == 0:
            import wandb
            run_handle = wandb.init(entity=settings['wandb_entity'], project=settings['wandb_project'],
                                    name=settings['run_name'], group=settings['group'], job_type='training',
                                    tags=(['distributed_training', 'memory_and_sharding', method,
                                           'ckpt_on' if settings['checkpoint_blocks'] else 'ckpt_off',
                                           'no_sync_on' if settings['no_sync'] else 'no_sync_off']
                                          if part_c else None),
                                    config={**settings, 'parameter_count': parameter_count,
                                            'gpu': torch.cuda.get_device_name(),
                                            'compute_dtype': 'bfloat16', 'parameter_dtype': 'float32'})
            # Validation occurs only at evaluation updates. An explicit axis
            # keeps those sparse values anchored to the optimizer update.
            run_handle.define_metric('optimizer_update')
            for namespace in ('train', 'val', 'data', 'perf', 'profile'):
                run_handle.define_metric(f'{namespace}/*', step_metric='optimizer_update')
        ids = microbatch_ids(rank, world, config.grad_accum_steps, settings['duplicate'])
        timed_seconds = timed_tokens = 0.0
        cumulative_tokens = 0
        last_metrics = {}
        validation_curve = []
        train_curve = []
        timed_updates = 0
        rank_peaks = [0] * world
        profiles = []
        for update in range(1, config.max_steps + 1):
            phase = 'train'
            # Replaying seeds across A1/A2 preserves the global batch, including
            # masked-pack document boundaries. No rank-dependent model seed.
            batches = [make_batch(corpus, mode=config.mode, batch_size=config.micro_batch_size,
                                  seq_len=config.seq_len, sql_fraction=config.sql_fraction,
                                  pad_id=identity['pad_id'], rng=batch_rng(config.seed, update, i)) for i in ids]
            # Hash actual input IDs to expose identical sampled rank data.
            # A3 should deliberately repeat hashes. This is an initial-data
            # diagnostic, not proof of dataset-wide uniqueness/no leakage.
            if update == 1:
                digest = hashlib.sha256(b''.join(b.input_ids.numpy().tobytes() for b in batches)).hexdigest()
                hashes = [None] * world
                if distributed:
                    dist.all_gather_object(hashes, digest)
                else:
                    hashes[0] = digest
                if rank == 0:
                    run_handle.summary['first_update_rank_batch_hashes'] = hashes
                    print(json.dumps({'first_update_rank_batch_hashes': hashes,
                                      'duplicate_data_intentional': settings['duplicate']}), flush=True)
            lr = learning_rate(update, config)
            for group in optimizer.param_groups:
                group['lr'] = lr
            # Clear ONCE per optimizer update. Clearing inside the microbatch
            # loop would destroy gradient accumulation. None avoids explicitly
            # zero-filling each gradient tensor before its next contribution.
            optimizer.zero_grad(set_to_none=True)
            if distributed:
                dist.barrier()
            # CUDA kernels are asynchronous with the CPU. Complete earlier
            # work before starting the clock; the barrier aligns worker starts.
            # Reset peak counters without discarding cached allocator storage.
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            profiled = update == settings.get('profile_update')
            if profiled:
                phase = 'profiled_train'
            with profile_update(profiled) as profiler:
                local_loss_sum, counts = accumulated_backward(
                    model, batches, device, world_size=world, no_sync=settings['no_sync'],
                    loss_chunk_size=config.loss_chunk_size)
                # Global clipping includes every shard before any rank updates.
                if sharded:
                    norm = model.clip_grad_norm_(config.grad_clip)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
                optimizer.step()
                torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started, device=device)
            memory = torch.tensor([torch.cuda.max_memory_allocated(device),
                                   torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.float64)
            if part_c:
                rank_memory = [torch.empty_like(memory) for _ in range(world)]
                dist.all_gather(rank_memory, memory)
                if not profiled:
                    rank_peaks = [max(peak, measured[0].item())
                                  for peak, measured in zip(rank_peaks, rank_memory)]
            if profiled:
                profiles = [None] * world
                dist.all_gather_object(profiles, export_profile(profiler, output, rank))
            if distributed:
                # A global update cannot finish before the slowest rank.
                # SUM the loss NUMERATOR, MAX time/memory; averaging rank
                # mean losses would be wrong when target counts differ.
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            # Scalar reads occur only at the update boundary. The measured step
            # includes H2D copies, forward/backward, count collective and AdamW;
            # excludes sampling, barrier, metric collectives, validation and IO.
            targets, tokens, sql_tokens = counts.tolist()
            cumulative_tokens += tokens
            seconds = elapsed.item()
            if not profiled and update > settings['benchmark_warmup']:
                timed_updates += 1
                timed_seconds += seconds
                timed_tokens += tokens
            # Slots include PAD; tokens excludes PAD; targets further excludes
            # masked/unpredictable positions. Keep these meanings distinct.
            # Rates count all rank occurrences (A3 still repeats actual data).
            slots = config.micro_batch_size * config.grad_accum_steps * world * config.seq_len
            metrics = {'train/loss': (local_loss_sum / counts[0]).item(),
                       'train/grad_norm': norm.item(), 'train/lr': lr,
                       'perf/step_seconds': seconds,
                       'perf/global_non_pad_tokens_per_sec': tokens / seconds,
                       'perf/global_raw_tokens_per_sec': slots / seconds,
                       'perf/cumulative_non_pad_tokens': cumulative_tokens,
                       'perf/padding_waste_pct': 100 * (slots - tokens) / slots,
                       'data/global_predicted_tokens': targets,
                       'data/observed_sql_token_fraction': sql_tokens / tokens,
                       'perf/peak_allocated_gib_max_rank': memory[0].item() / 1024**3,
                       'perf/peak_reserved_gib_max_rank': memory[1].item() / 1024**3}
            if part_c:
                for gpu, measured in enumerate(rank_memory):
                    metrics[f'perf/gpu_{gpu}_peak_allocated_gib'] = measured[0].item() / 1024**3
                if profiled:
                    for gpu, profile in enumerate(profiles):
                        metrics.update({f'profile/gpu_{gpu}/{key}': value for key, value in profile.items()})
                    for key in ('perf/step_seconds', 'perf/global_non_pad_tokens_per_sec',
                                'perf/global_raw_tokens_per_sec'):
                        metrics.pop(key)
            if update % config.eval_every == 0 or update == config.max_steps:
                phase = 'validation'
                # FSDP validation replays the fixed sample on every rank to
                # enter matching gathers. DDP can evaluate on rank 0 alone.
                if sharded or rank == 0:
                    metrics.update(evaluate_sql(model if sharded else raw_model, validation,
                                                config, identity['pad_id'], device))
                if rank == 0:
                    # Store only sampled points so the final full-corpus
                    # metric cannot appear as a jump in the same series.
                    validation_curve.append({'update': update, 'loss': metrics['val/sql_loss']})
                    if update == config.max_steps and not part_c:
                        metrics.update(evaluate_sql(raw_model, validation, config, identity['pad_id'], device, full=True))
                if distributed:
                    # Rank 1 must not start its next training collective while
                    # rank 0 is still evaluating without DDP participation.
                    dist.barrier()
            if rank == 0:
                train_curve.append({'update': update, 'loss': metrics['train/loss']})
                history = {'optimizer_update': update, **metrics}
                if update == config.max_steps:
                    import wandb
                    # Explicitly label charts with the quantity and update
                    # axis. Plot sparse validation at the updates it ran.
                    for key, title, curve in (
                        ('training_loss', 'Training next-token loss', train_curve),
                        ('sql_validation_loss', 'Sampled SQL validation loss', validation_curve),
                    ):
                        history[f'charts/{key}'] = wandb.plot.line_series(
                            xs=[[point['update'] for point in curve]],
                            ys=[[point['loss'] for point in curve]],
                            keys=[title], title=title, xname='Optimizer update',
                        )
                    if part_c:
                        # A one-case suite has no comparison run. Its two GPU
                        # profiles are available directly on the training run.
                        gpu_table = wandb.Table(columns=['GPU', 'NCCL (ms)', 'Other kernels (ms)', 'Overlap (ms)'])
                        for gpu, profile in enumerate(profiles):
                            gpu_table.add_data(f'GPU {gpu}', profile['nccl_kernel_ms'],
                                               profile['other_kernel_ms'], profile['overlap_ms'])
                        history['charts/gpu_profile'] = gpu_table
                run_handle.log(history, step=update)
                print(json.dumps({'update': update, **metrics}), flush=True)
                last_metrics = metrics
        phase = 'reporting'
        if rank == 0:
            if part_c:
                import wandb
                traces = wandb.Artifact(f'{run_handle.id}-gpu-traces', type='gpu-profile')
                for gpu in range(world):
                    traces.add_file(str(output / f'trace-rank-{gpu}.json'))
                run_handle.log_artifact(traces)
            else:
                # A/B retains the snapshots needed for its weight comparison.
                temporary = output / 'final.tmp'
                torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'update': config.max_steps, 'settings': settings,
                            'model_config': asdict(raw_model.config)}, temporary)
                os.replace(temporary, output / 'final.pt')
            summary = {'run_name': settings['run_name'], 'parameter_count': parameter_count,
                       'steady_global_non_pad_tokens_per_sec': timed_tokens / timed_seconds if timed_seconds else None,
                       'timed_updates': timed_updates,
                       'wandb_url': run_handle.url}
            if part_c:
                summary['oom'] = False
            # W&B history already contains the train/val/perf/profile series.
            # Nested final values in the summary look like plottable series
            # in the workspace but have no history. Keep them in the JSON.
            run_handle.summary.update(summary)
            result = {**summary, 'last_metrics': last_metrics, 'validation_curve': validation_curve}
            if part_c:
                result.update(rank_peak_allocated_bytes=rank_peaks if config.max_steps > 1 else [None] * world,
                              profile_by_rank=profiles, train_curve=train_curve)
            (output / 'result.json').write_text(json.dumps(result, indent=2))
        if distributed:
            dist.barrier()
    except torch.cuda.OutOfMemoryError:
        if part_c:
            (output / f'oom-rank-{rank}.json').write_text(json.dumps({
                'update': update, 'phase': phase, 'parameter_count': parameter_count,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            }))
        raise
    finally:
        # Close tracking and distributed resources even when a worker raises.
        # torchrun detects a failed worker and terminates its peers; a missing
        # rank must not leave a later experiment falsely reported as successful.
        if run_handle is not None:
            run_handle.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True, help='JSON settings written by Modal parent')
    run(json.loads(parser.parse_args().settings.read_text()))

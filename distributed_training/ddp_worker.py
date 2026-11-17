"""Torchrun worker: one process per GPU, one complete model copy per process.

Launched by modal_ddp_parts_ab.py, not directly by `modal run`. Paths and
hyperparameters arrive in an explicit JSON file shared by all local ranks.
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
from distributed_training.ddp_core import accumulated_backward, batch_rng, microbatch_ids
from model.transformer_35m import Transformer, TransformerConfig
from train_pretrain import TrainConfig, learning_rate
from distributed_training.sql_validation import load_training_and_sql_validation, evaluate_sql


def run(settings: dict) -> None:
    """Run one labeled A/B experiment, using the same code path for all cases.

    settings contains train/model dataclass fields plus world_size, duplicate,
    no_sync, run_name, output_dir, benchmark_warmup, wandb_entity and project.
    Results include global useful-token throughput, per-rank peak memory,
    filtered validation, and a final checkpoint. No implicit checkpoint resume:
    every experiment starts from the same seed and a new output directory.
    """
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != settings['world_size']:
        raise ValueError('torchrun world size differs from experiment settings')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('These experiments require CUDA BF16 support')
    distributed = world > 1
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    run_handle = None
    try:
        config = TrainConfig(**settings['train'])
        output = Path(settings['output_dir'])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            tokenizer, corpus, validation = load_training_and_sql_validation()
            identity = {'vocab_size': len(tokenizer), 'pad_id': tokenizer.pad_token_id,
                        'eos_id': tokenizer.eos_token_id}
        else:
            identity = None
        if distributed:
            objects = [identity]
            dist.broadcast_object_list(objects, src=0, device=device)
            identity = objects[0]
        if rank != 0:
            # Only rank 0 scans validation and filters overlap. Each worker has
            # its own read-only memmaps of training, with identical row indexes.
            corpus = Corpus(Path('/vol/pretrain_data'), 'train', identity['eos_id'])
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        raw_model = Transformer(TransformerConfig(vocab_size=identity['vocab_size'],
                                                 max_seq_len=config.seq_len,
                                                 **settings['model'])).to(device)
        model = (DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False,
                     gradient_as_bucket_view=True) if distributed else raw_model)
        # BF16 autocast changes eligible compute; stored weights/gradients and
        # Adam moments stay FP32. Bucket views avoid a separate gradient copy.
        decay = [p for p in raw_model.parameters() if p.ndim >= 2]
        no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': config.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=config.peak_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        parameter_count = sum(p.numel() for p in raw_model.parameters())
        if rank == 0:
            import wandb
            run_handle = wandb.init(entity=settings['wandb_entity'], project=settings['wandb_project'],
                                    name=settings['run_name'], group=settings['group'],
                                    config={**settings, 'parameter_count': parameter_count,
                                            'gpu': torch.cuda.get_device_name(),
                                            'compute_dtype': 'bfloat16', 'parameter_dtype': 'float32'})
        ids = microbatch_ids(rank, world, config.grad_accum_steps, settings['duplicate'])
        timed_seconds = timed_tokens = 0.0
        cumulative_tokens = 0
        last_metrics = {}
        for update in range(1, config.max_steps + 1):
            # Replaying seeds across A1/A2 preserves the global batch, including
            # masked-pack document boundaries. No rank-dependent model seed.
            batches = [make_batch(corpus, mode=config.mode, batch_size=config.micro_batch_size,
                                  seq_len=config.seq_len, sql_fraction=config.sql_fraction,
                                  pad_id=identity['pad_id'], rng=batch_rng(config.seed, update, i)) for i in ids]
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
            optimizer.zero_grad(set_to_none=True)
            if distributed:
                dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            local_loss_sum, counts = accumulated_backward(
                model, batches, device, world_size=world, no_sync=settings['no_sync'],
                loss_chunk_size=config.loss_chunk_size)
            # Clip AFTER accumulation and DDP synchronization, so every rank
            # clips the same global gradient and applies the same AdamW step.
            norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started, device=device)
            memory = torch.tensor([torch.cuda.max_memory_allocated(device),
                                   torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            # Scalar reads occur only at the update boundary. The measured step
            # includes H2D copies, forward/backward, count collective and AdamW;
            # excludes sampling, barrier, metric collectives, validation and IO.
            targets, tokens, sql_tokens = counts.tolist()
            cumulative_tokens += tokens
            seconds = elapsed.item()
            if update > settings['benchmark_warmup']:
                timed_seconds += seconds
                timed_tokens += tokens
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
            if update % config.eval_every == 0 or update == config.max_steps:
                if rank == 0:
                    # Use the UNWRAPPED model: other ranks wait at a barrier;
                    # a DDP forward here could hang on distributed collectives.
                    metrics.update(evaluate_sql(raw_model, validation, config, identity['pad_id'], device))
                    if update == config.max_steps:
                        metrics.update(evaluate_sql(raw_model, validation, config, identity['pad_id'], device, full=True))
                if distributed:
                    dist.barrier()
            if rank == 0:
                run_handle.log(metrics, step=update)
                print(json.dumps({'update': update, **metrics}), flush=True)
                last_metrics = metrics
        if rank == 0:
            # Save only on rank 0. The Modal parent commits after torchrun exits.
            # This is a final experiment snapshot, not an automatic-resume API.
            temporary = output / 'final.tmp'
            torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'update': config.max_steps, 'settings': settings,
                        'model_config': asdict(raw_model.config)}, temporary)
            os.replace(temporary, output / 'final.pt')
            summary = {'run_name': settings['run_name'], 'parameter_count': parameter_count,
                       'steady_global_non_pad_tokens_per_sec': timed_tokens / timed_seconds if timed_seconds else None,
                       'timed_updates': max(0, config.max_steps - settings['benchmark_warmup']),
                       'wandb_url': run_handle.url, 'last_metrics': last_metrics}
            run_handle.summary.update(summary)
            (output / 'result.json').write_text(json.dumps(summary, indent=2))
        if distributed:
            dist.barrier()
    finally:
        if run_handle is not None:
            run_handle.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True, help='JSON settings written by Modal parent')
    run(json.loads(parser.parse_args().settings.read_text()))

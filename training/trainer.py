"""Shared single-GPU data → loss → update loop, independent of experiment launchers.

Experiment modules import run_training and supply a TrainConfig plus output and
tracking settings. Launch a particular experiment from its launch.py.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import time
import uuid

from training.errors import NumericalDivergence
from training.checkpointing import restore_checkpoint, restore_branch, save_checkpoint
from training.dataset import dataset_identity
from training.tracking import curve_payload, define_metrics, log_update
from training.config import DATA_DIR, TrainConfig, _load_data_and_tokenizer, learning_rate, volume

CHECKPOINT_DIR = Path('/vol/checkpoints/packing')


def _evaluate(model, corpus, config: TrainConfig, pad_id: int, device) -> dict[str, float]:
    """Evaluate fixed standalone documents, using the same target-weighted loss.

    The seed and schedule comparison fixes validation_seed across training seeds.
    A seed-dependent validation sample would mix evaluation variation into the
    measured training noise floor.
    Each corpus has its own generator, independent of the training sample stream.
    """
    import torch
    from data.dataloader import make_batch
    model.eval()
    results = {}
    seed = config.seed if config.validation_seed is None else config.validation_seed
    for kind, offset in (('english', 10_000), ('sql', 20_000)):
        rng = random.Random(seed + offset)
        total_loss, total_targets = torch.zeros((), device=device), 0
        with torch.inference_mode():
            for _ in range(config.val_batches):
                batch = make_batch(corpus, mode='pad', batch_size=1, seq_len=config.seq_len,
                                   sql_fraction=config.sql_fraction, pad_id=pad_id, rng=rng,
                                   validation_kind=kind)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs('pad', device), loss_chunk_size=config.loss_chunk_size)
                total_loss += loss * batch.predicted_tokens
                total_targets += batch.predicted_tokens
        results[f'val/{kind}_loss'] = float(total_loss / total_targets)
    model.train()
    return results


def _checkpoint_path(run_name: str) -> Path:
    """Keep each run's checkpoint in a safe, distinct directory."""
    allowed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.'
    if not run_name or run_name in {'.', '..'} or any(char not in allowed for char in run_name):
        raise ValueError('run_name may contain only letters, digits, hyphens, dots, underscores')
    return CHECKPOINT_DIR / run_name / 'latest.pt'


def run_training(config: TrainConfig, wandb_entity: str, wandb_project: str, run_name: str, *,
         output_dir: Path | None = None, group: str | None = None, title: str | None = None,
         expected_dataset: dict | None = None, milestone_updates: tuple[int, ...] = (),
         stop_after_update: int | None = None, x_axis: str = 'optimizer_update',
         probe_every: int = 0, probe_seed: int = 23456,
         branch_from: Path | None = None, record_sample_hashes: bool = False) -> dict:
    """Train/resume one case; each microbatch contributes its valid-target share.

    output_dir isolates suite checkpoints; None preserves historical packing paths.
    expected_dataset pins the published build across workers. milestone_updates
    retains selected snapshots in addition to latest.pt. stop_after_update pauses
    a bounded correctness run without changing max_steps or its LR trajectory.
    x_axis selects update- or token-based plots without changing optimization.
    probe_every enables a fixed training-batch loss diagnostic; 0 disables it.
    probe_seed uses an independent sampler. The recipe is checkpointed and must
    remain identical on resume. Enabled probes also retain actual update vectors
    at milestone_updates, for later directional curvature/branching work.
    The result/history/config files are durable outputs, not fabricated metrics.
    branch_from explicitly forks a retained checkpoint into a new output directory,
    permitting only constant-LR schedule/length changes. Ordinary resume stays
    strict. record_sample_hashes verifies actual future input equality across arms.
    """
    if x_axis not in {'optimizer_update', 'tokens_seen'}:
        raise ValueError('x_axis must be optimizer_update or tokens_seen')
    if (isinstance(probe_every, bool) or not isinstance(probe_every, int) or probe_every < 0
            or isinstance(probe_seed, bool) or not isinstance(probe_seed, int) or probe_seed < 0):
        raise ValueError('Probe interval and seed must be nonnegative integers')
    import torch
    import wandb
    from data.dataloader import make_batch
    from training.optimizers import build_optimizer
    from model.transformer_35m import Transformer, TransformerConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run needs a CUDA GPU with bf16 support')
    if not wandb_entity:
        raise ValueError('Pass --wandb-entity to record a reproducible run')
    final_update = config.max_steps if stop_after_update is None else stop_after_update
    if not 1 <= final_update <= config.max_steps:
        raise ValueError('stop_after_update must be within the training run')
    volume.reload()
    path = _checkpoint_path(run_name) if output_dir is None else output_dir / 'latest.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(path, map_location='cpu', weights_only=False) if path.is_file() else None
    forking = branch_from is not None and state is None
    if branch_from is not None:
        if output_dir is None or path.parent.resolve() == branch_from.parent.resolve():
            raise ValueError('A branch needs its own output directory')
        if forking:
            state = torch.load(branch_from, map_location='cpu', weights_only=False)
        elif (state.get('branch_origin') or {}).get('checkpoint') != str(branch_from):
            raise ValueError('Resume branch source differs from the checkpoint')
    probe_recipe = {'every': probe_every, 'seed': probe_seed} if probe_every else {}
    if state and not forking and state.get('probe_recipe', {}) != probe_recipe:
        raise ValueError('Fixed probe recipe differs from the checkpoint')
    branch_origin = state.get('branch_origin') if state and not forking else None
    if forking:
        if not probe_every or state.get('probe_recipe', {}).get('seed') != probe_seed:
            raise ValueError('Branch must retain the saved fixed-probe seed')
        branch_origin = {'checkpoint': str(branch_from), 'update': state['update'],
                         'tokens_seen': state['cumulative_tokens'], 'source_config': state['config']}
    if state is not None and not state.get('grouped_sql_dir'):
        raise ValueError('Checkpoint predates Grouped SQL; use a new run name')
    saved_dir = Path(state['grouped_sql_dir']) if state else None
    identity = dataset_identity(DATA_DIR, saved_dir)
    if expected_dataset is not None and identity != expected_dataset:
        raise ValueError('Published dataset changed after this suite was planned')
    if state and state.get('dataset_identity') not in (None, identity):
        raise ValueError('Published dataset differs from the checkpoint')
    grouped_sql_dir = Path(identity['sql_dir'])
    tokenizer, train, validation = _load_data_and_tokenizer(grouped_sql_dir, reload_volume=False)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    rng = random.Random(config.seed)
    device = torch.device('cuda')
    # Construction uses its own Python sampler and never advances rng. Saving
    # the CPU batch lets future probes use exactly these token IDs and masks.
    probe = None
    if probe_every:
        if branch_from is not None:
            saved_probe = torch.load(branch_from.parent / 'fixed_probe.pt', map_location='cpu', weights_only=False)
            if saved_probe['dataset_identity'] != identity:
                raise ValueError('Saved probe data differs from the branch data')
            probe = saved_probe['batch']
        else:
            probe = [make_batch(train, mode=config.mode, batch_size=1,
                            seq_len=config.seq_len, sql_fraction=config.sql_fraction,
                            pad_id=tokenizer.pad_token_id, rng=random.Random(probe_seed + offset),
                            validation_kind=kind)
                 for kind, offset in (('english', 10000), ('sql', 20000))]
        torch.save({'batch': probe, 'recipe': probe_recipe, 'dataset_identity': identity},
                   path.parent / 'fixed_probe.pt')
    model = Transformer(TransformerConfig(vocab_size=len(tokenizer), max_seq_len=config.seq_len,
                                        initializer_range=config.initializer_range)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = build_optimizer(model, config)
    start_update, cumulative_tokens, wandb_id, history = 0, 0, uuid.uuid4().hex, []
    sample_hashes = []
    if forking:
        restore_branch(state, model, optimizer, config, rng)
    elif state:
        restore_checkpoint(state, model, optimizer, config, rng)
        if config.optimizer == 'adamw':  # Older checkpoints may predate fused AdamW.
            for param_group in optimizer.param_groups:
                param_group['fused'] = True
        start_update, cumulative_tokens = state['update'], state['cumulative_tokens']
        wandb_id, history = state['wandb_id'], state.get('history', [])
        sample_hashes = state.get('sample_hashes', [])
        if record_sample_hashes and len(sample_hashes) != start_update:
            raise ValueError('Branch replay lacks its sampled-input hash prefix')
    del state  # CPU checkpoint tensors need not remain live during training.
    settings = {**asdict(config), 'parameter_count': parameter_count, 'dataset_identity': identity,
                'gpu': torch.cuda.get_device_name(), 'compute_dtype': 'bfloat16',
                'parameter_dtype': 'float32', 'checkpoint_name': run_name, 'probe_recipe': probe_recipe,
                'branch_origin': branch_origin}
    initial_probe_loss = None
    if branch_origin is not None:
        from training.probes import fixed_batch_loss
        initial_probe_loss = branch_origin.get('initial_probe_loss')
        if initial_probe_loss is None:
            initial_probe_loss = fixed_batch_loss(model, probe, config, device)
            branch_origin['initial_probe_loss'] = initial_probe_loss
        settings['initial_probe_loss'] = initial_probe_loss
    (path.parent / 'settings.json').write_text(json.dumps(settings, indent=2))
    print(json.dumps({'run': run_name, 'resume_after_update': start_update, **settings}), flush=True)
    # Rewrite only the committed prefix on resume. An interrupted attempt may
    # have written later metric rows without saving the corresponding weights.
    with (path.parent / 'metrics.jsonl').open('w') as output, wandb.init(
        entity=wandb_entity, project=wandb_project, name=title or run_name,
        group=group or f'Packing · {identity["corpus_id"][:8]} · {config.seq_len} tokens',
        job_type='training', id=wandb_id, resume='allow', config=settings,
        settings=wandb.Settings(init_timeout=120),
    ) as run:
        define_metrics(run, step_metric=x_axis)
        for row in history:
            output.write(json.dumps(row) + '\n')
        torch.cuda.reset_peak_memory_stats()
        for update in range(start_update + 1, final_update + 1):
            started = time.perf_counter()
            lr = learning_rate(update, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
            optimizer.zero_grad(set_to_none=True)
            before = ({name: p.detach().clone() for name, p in model.named_parameters()}
                      if probe_every and update in milestone_updates else None)
            batches = [make_batch(train, mode=config.mode, batch_size=config.micro_batch_size,
                                  seq_len=config.seq_len, sql_fraction=config.sql_fraction,
                                  pad_id=tokenizer.pad_token_id, rng=rng)
                       for _ in range(config.grad_accum_steps)]
            sample_hash = None
            if record_sample_hashes:
                from training.probes import batch_fingerprint
                sample_hash = batch_fingerprint(batches)
            target_count = sum(batch.predicted_tokens for batch in batches)
            weighted_loss = torch.zeros((), device=device)
            for batch in batches:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, loss = model(**batch.model_inputs(config.mode, device),
                                    loss_chunk_size=config.loss_chunk_size)
                # Example: a microbatch with 100 targets gets twice the weight
                # of one with 50. Averaging batch means equally would be wrong.
                weight = batch.predicted_tokens / target_count
                (loss * weight).backward()
                weighted_loss += loss.detach() * weight
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                          error_if_nonfinite=True)
            except RuntimeError as error:
                if 'non-finite' not in str(error):
                    raise  # Do not label OOMs or unrelated runtime bugs as LR divergence.
                raise NumericalDivergence(update, str(error)) from error
            optimizer.step()
            direction = None
            if before is not None:
                from training.probes import update_direction
                direction = update_direction(model, before)
                del before
            # One combined scalar read waits for completed GPU work; none occurs
            # between microbatches. Timing excludes validation, checkpoint IO, W&B.
            logged_loss, logged_grad_norm = torch.stack((weighted_loss, grad_norm)).tolist()
            if not math.isfinite(logged_loss):
                raise NumericalDivergence(update, 'nonfinite training loss')
            elapsed = time.perf_counter() - started
            real_tokens = sum(batch.non_pad_tokens for batch in batches)
            sql_tokens = sum(batch.sql_tokens for batch in batches)
            cumulative_tokens += real_tokens
            slots = config.micro_batch_size * config.seq_len * config.grad_accum_steps
            metrics = {'tokens_seen': cumulative_tokens, 'data/predicted_tokens': target_count,
                       'train/loss': logged_loss, 'train/grad_norm': logged_grad_norm, 'train/lr': lr,
                       'data/observed_sql_token_fraction': sql_tokens / real_tokens,
                       'perf/raw_tokens_per_sec': slots / elapsed,
                       'perf/effective_non_pad_tokens_per_sec': real_tokens / elapsed,
                       'perf/padding_waste_pct': 100 * (slots - real_tokens) / slots,
                       'perf/cumulative_non_pad_tokens': cumulative_tokens,
                       'perf/peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                       'perf/peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3}
            if config.optimizer == 'muon':
                metrics['train/companion_lr'] = optimizer.adamw.param_groups[0]['lr']
            if probe_every:
                metrics['train/clipped'] = int(logged_grad_norm > config.grad_clip)
                if update == 1 or update % probe_every == 0 or update == final_update:
                    from training.probes import fixed_batch_loss
                    metrics['probe/fixed_loss'] = fixed_batch_loss(model, probe, config, device)
                    if not math.isfinite(metrics['probe/fixed_loss']):
                        raise NumericalDivergence(update, 'nonfinite fixed-probe loss')
            evaluate = update % config.eval_every == 0 or update == final_update
            if evaluate:
                metrics.update(_evaluate(model, validation, config, tokenizer.pad_token_id, device))
                if any(not math.isfinite(metrics[key]) for key in ('val/sql_loss', 'val/english_loss')):
                    raise NumericalDivergence(update, 'nonfinite validation loss')
            log_update(run, update, metrics, history, output)
            if record_sample_hashes:
                sample_hashes.append(sample_hash)
                # Also retain successful input identities if a later numerical
                # failure exits before the next full-state checkpoint.
                temporary = path.parent / 'sample_hashes.tmp'
                temporary.write_text(json.dumps(sample_hashes))
                temporary.replace(path.parent / 'sample_hashes.json')
            if evaluate or update in milestone_updates:
                output.flush()
                save_checkpoint(path, model, optimizer, config, update, rng, cumulative_tokens,
                                wandb_id, grouped_sql_dir, history=history, dataset=identity,
                                update_direction=direction, probe_recipe=probe_recipe,
                                branch_origin=branch_origin, sample_hashes=sample_hashes)
                if update in milestone_updates:
                    shutil.copyfile(path, path.with_name(f'step-{update:06d}.pt'))
                volume.commit()
            print(json.dumps({'update': update, **metrics}), flush=True)
        if history:
            run.log(curve_payload(wandb, [history], [title or run_name], x_key=x_axis))
        last_metrics = history[-1] if history else {}
        run.summary.update({'final/sql_loss': last_metrics.get('val/sql_loss'),
                            'final/english_loss': last_metrics.get('val/english_loss'),
                            'completed_updates': final_update})
        result = {'run': run_name, 'updates': final_update, 'completed': final_update == config.max_steps,
                  'parameter_count': parameter_count, 'checkpoint': str(path), 'wandb_url': run.url,
                  'wandb_path': run.path, 'dataset_identity': identity,
                  'config': asdict(config), 'history': history, 'probe_recipe': probe_recipe,
                  'branch_origin': branch_origin, 'sample_hashes': sample_hashes,
                  'initial_probe_loss': initial_probe_loss}
    # The W&B context has finished/flushed before an API verifier reads the run.
    (path.parent / 'result.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


# Preserve existing launch modules while giving the engine a public name.
_run = run_training

"""Optimizer choices for the shared single-GPU update loop."""
import torch


class MuonWithAdamW:
    """Step two disjoint parameter sets together and checkpoint both histories."""

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        # Look up current groups: load_state_dict can replace group dictionaries.
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, *, set_to_none):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])


def build_optimizer(model, config):
    """Map a Transformer and TrainConfig to the optimizer used after accumulation.

    Muon transforms only hidden 2-D weights. The shared embedding/output matrix
    and all 1-D norm weights use a companion AdamW with its own peak LR.
    """
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    vectors = [p for p in model.parameters() if p.ndim < 2]
    groups = [{'params': matrices, 'weight_decay': config.weight_decay},
              {'params': vectors, 'weight_decay': 0.0}]
    if config.optimizer == 'momentum_sgd':
        return torch.optim.SGD(groups, lr=config.peak_lr, momentum=0.9)

    fused = model.token_embedding.weight.is_cuda
    if config.optimizer == 'muon':
        embedding = model.token_embedding.weight
        hidden = [p for p in matrices if p is not embedding]
        muon = torch.optim.Muon(hidden, lr=config.peak_lr, weight_decay=config.weight_decay)
        groups[0]['params'] = [embedding]
        for group in groups:
            group['lr_scale'] = config.companion_peak_lr / config.peak_lr
        adamw = torch.optim.AdamW(groups, lr=config.companion_peak_lr, betas=(.9, .95), fused=fused)
        return MuonWithAdamW(muon, adamw)
    return torch.optim.AdamW(groups, lr=config.peak_lr, betas=(.9, .95), fused=fused)

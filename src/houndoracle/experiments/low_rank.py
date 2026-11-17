"""Small local attention adapters and an explicitly supervised contrastive objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("Expected a linear layer and positive rank")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values):
        return self.base(values) + F.linear(F.linear(values, self.lora_A), self.lora_B)


def attach(model, *, rank=8, last_layers=4):
    if (
        model.config.model_type != "deberta-v2"
        or not 1 <= last_layers <= model.config.num_hidden_layers
    ):
        raise ValueError("Supported DeBERTa architecture/layer count required")
    model.requires_grad_(False)
    targets = [
        f"deberta.encoder.layer.{i}.attention.self.{projection}"
        for i in range(model.config.num_hidden_layers - last_layers, model.config.num_hidden_layers)
        for projection in ["query_proj", "value_proj"]
    ]
    for name in targets:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LowRankLinear(getattr(parent, child), rank))
    model.classifier.requires_grad_(True)
    return targets


def objective(logits, labels, *, pair_weight=0.2, margin=1.0):
    if (
        logits.ndim != 2
        or logits.shape[1] != 3
        or logits.shape[0] != labels.numel()
        or labels.numel() % 2
    ):
        raise ValueError("Require adjacent contrasting pairs of three-class logits")
    if torch.any(labels[::2] == labels[1::2]):
        raise ValueError("Each pair must have different independently supplied labels")
    ce = F.cross_entropy(logits, labels)
    a, b = logits[::2], logits[1::2]
    ya, yb = labels[::2, None], labels[1::2, None]
    separation = ((a.gather(1, ya) - a.gather(1, yb)) + (b.gather(1, yb) - b.gather(1, ya))) / 2
    ranking = F.softplus(margin - separation).mean()
    return ce + pair_weight * ranking, ce, ranking


def trainable_state(model):
    return {
        name: p.detach().cpu().contiguous()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def restore_trainable(model, state):
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected) or any(
        state[n].shape != p.shape or state[n].dtype != p.dtype for n, p in expected.items()
    ):
        raise ValueError("Adapter checkpoint does not match the complete expected parameter schema")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name])

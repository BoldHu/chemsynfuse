"""Molecular pretraining step on externally tokenized molecular views."""
import torch
from torch.nn import functional as F
from .constants import PAD
from .molecular import molecular_contrastive


def pretraining_step(model, optimizer, first_view, second_view, target_tokens,
                     identities, contrastive_weight=.1, temperature=.1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, a, b = model(first_view, second_view, target_tokens)
    reconstruction = F.cross_entropy(logits.flatten(0, 1), target_tokens[:, 1:].flatten(), ignore_index=PAD)
    contrastive, _ = molecular_contrastive(a, b, identities, temperature)
    loss = reconstruction + contrastive_weight * contrastive
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite pretraining objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite pretraining gradient')
    optimizer.step()
    return {'loss': float(loss.detach()), 'reconstruction': float(reconstruction.detach()),
            'contrastive': float(contrastive.detach())}

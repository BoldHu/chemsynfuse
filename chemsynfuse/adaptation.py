"""Conditional domain alignment with explicit membership, support and GRL semantics.

No target-label argument exists. Supervised CE/MSE has no class weights. Lambda
acts once in the reused GRL; domain heads minimize ordinary positive BCE.
"""
import copy
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .gradient import ReverseLayerF

def effective_size(weights):
    w = weights.detach()
    return float(w.sum().square() / w.square().sum().clamp_min(1e-12))

def classification_members(source_y, teacher_probabilities, reliability=None):
    p = teacher_probabilities.detach()
    if p.ndim != 2 or not torch.isfinite(p).all() or (p < 0).any() or (not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-05)):
        raise ValueError('Normalized finite teacher probabilities required')
    source = F.one_hot(source_y.long(), p.shape[1]).float()
    r = torch.ones(len(p), device=p.device) if reliability is None else reliability.detach()
    if r.shape != (len(p),) or not torch.isfinite(r).all() or (r < 0).any() or (r > 1).any():
        raise ValueError('Invalid target reliability')
    return (source.detach(), (p * r[:, None]).detach())

class RegressionNeighborhoods:
    """Source-only standardized quantile anchors; continuous head stays continuous."""

    def __init__(self, state):
        self.state = state

    def standardize(self, raw):
        return (raw - self.state['mean']) / self.state['scale']

    def members(self, standardized_prediction):
        z = standardized_prediction.detach().flatten()
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite continuous prediction')
        a = torch.tensor(self.state['anchors'], device=z.device, dtype=z.dtype)
        inside = (z >= self.state['minimum']) & (z <= self.state['maximum'])
        safe_z = torch.where(inside, z, torch.zeros_like(z))
        distance = (safe_z.double()[:, None] - a.double()).square()
        distance = distance - distance.min(-1, keepdim=True).values
        logits = -0.5 * (distance / self.state['bandwidth']) / self.state['bandwidth']
        q = torch.softmax(logits, -1).to(z.dtype) * inside[:, None]
        return (q, dict(outside_source_range=int((~inside).sum()), accepted=int(inside.sum())))

class EMATeacher(nn.Module):

    def __init__(self, student, decay=0.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('EMA decay outside [0,1)')
        self.model = copy.deepcopy(student).requires_grad_(False).eval()
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.decay = decay

    @torch.no_grad()
    def update(self, student):
        src = student.state_dict()
        for name, target in self.model.state_dict().items():
            if target.is_floating_point():
                target.mul_(self.decay).add_(src[name], alpha=1 - self.decay)
            else:
                target.copy_(src[name])
        self.updates.add_(1)
        self.model.eval()

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        return self.model(batch)

class FusionPredictor(nn.Module):
    """Continuous regression or ordinary classification on the prediction branch."""

    def __init__(self, fusion, outputs):
        super().__init__()
        self.fusion = fusion
        self.head = nn.Linear(fusion.dim, outputs)

    def forward(self, batch):
        return self.head(self.fusion(batch)['fused'])

def grl_schedule(progress, maximum=1.0):
    if not 0 <= progress <= 1 or not 0 <= maximum <= 1:
        raise ValueError('Progress/maximum outside [0,1]')
    return maximum * (2 / (1 + math.exp(-10 * progress)) - 1)

def teacher_targets(view_predictions, kind, confidence=0.8, max_disagreement=0.05, max_regression_std=0.1, temperature=1.0):
    """Views x batch x outputs; uncertainty uses predictions, never target truth.

    With one view uncertainty is unmeasured: explicit status, not fabricated zero
    evidence. Use two identity-preserving views when requiring consistency.
    """
    if temperature <= 0 or not 0 <= confidence <= 1 or max_disagreement < 0 or (max_regression_std < 0):
        raise ValueError('Invalid teacher filtering thresholds')
    views = view_predictions.detach()
    if views.ndim != 3 or views.shape[0] < 1 or (not torch.isfinite(views).all()):
        raise ValueError('Finite V,B,C teacher predictions required')
    multi = len(views) >= 2
    if kind == 'classification':
        prob = torch.softmax(views / temperature, -1)
        target = prob.mean(0)
        uncertainty = (prob - target[None]).abs().amax((0, 2))
        accepted = target.max(-1).values >= confidence
        if multi:
            accepted &= uncertainty <= max_disagreement
    elif kind == 'regression':
        if views.shape[-1] != 1:
            raise ValueError('Regression head must be continuous scalar')
        target = views.mean(0).flatten()
        uncertainty = views.squeeze(-1).std(0, unbiased=False)
        accepted = torch.ones_like(target, dtype=torch.bool)
        if multi:
            accepted &= uncertainty <= max_regression_std
    else:
        raise ValueError('Unknown prediction kind')
    return (target, accepted.float(), dict(views=len(views), uncertainty_measured=multi, accepted=int(accepted.sum()), total=len(accepted), uncertainty=uncertainty.cpu().tolist() if multi else None))

def pseudo_label_loss(student_prediction, target, reliability, kind, enabled=False):
    if not enabled:
        return student_prediction.sum() * 0
    r = reliability.detach()
    tolerance = 4 * torch.finfo(r.dtype).eps
    if not torch.isfinite(r).all() or (r < 0).any() or (r > 1 + tolerance).any():
        raise ValueError('Invalid reliability')
    r = r.clamp_max(1.0)
    if kind == 'classification':
        per_row = -(target.detach() * F.log_softmax(student_prediction, -1)).sum(-1)
    elif kind == 'regression':
        per_row = F.smooth_l1_loss(student_prediction.flatten(), target.detach().flatten(), reduction='none')
    else:
        raise ValueError('Unknown kind')
    return (per_row * r).sum() / r.sum().clamp_min(1e-12)

class PathStem(nn.Module):
    """Three real source-contribution tokens, shared over all local outcomes."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.source_type = nn.Parameter(torch.zeros(1, 3, dim))
        nn.init.normal_(self.source_type, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, paths, availability):
        if paths.ndim != 3 or paths.shape[1] != 3 or availability.shape != paths.shape[:2]:
            raise ValueError('Path stem requires three source tokens, never length-one attention')
        tokens = paths.masked_fill(~availability[:, :, None], 0.0)
        empty = ~availability.any(1)
        tokens = (tokens + self.source_type).masked_fill(~availability[:, :, None], 0.0)
        safe = availability.clone()
        safe[empty, 0] = True
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~safe, need_weights=False)
        attended = self.norm(tokens + attended).masked_fill(~availability[:, :, None], 0.0)
        result = attended.sum(1) / availability.sum(1, keepdim=True).clamp_min(1)
        return result.masked_fill(empty[:, None], 0.0)

@dataclass
class SupportPolicy:
    minimum_count: int = 2
    minimum_mass: float = 0.1
    minimum_effective: float = 2.0

    def __post_init__(self):
        if not all((math.isfinite(v) for v in (self.minimum_count, self.minimum_mass, self.minimum_effective))) or self.minimum_count < 1 or self.minimum_mass <= 0 or (self.minimum_effective < 1):
            raise ValueError('Positive support thresholds required')

    def check(self, weights):
        w = weights.detach()
        stats = dict(count=int((w > 0).sum()), mass=float(w.sum()), effective=effective_size(w))
        supported = stats['count'] >= self.minimum_count and stats['mass'] >= self.minimum_mass and (stats['effective'] + 1e-06 >= self.minimum_effective)
        return (supported, stats)

def pattern_ids(availability):
    if availability.ndim != 2 or availability.shape[1] != 3 or availability.dtype != torch.bool:
        raise ValueError('Three boolean source availability columns required')
    return (availability.long() * torch.tensor([1, 2, 4], device=availability.device)).sum(-1)

def conditional_domain_loss(source_logits, target_logits, source_members, target_members, source_availability, target_availability, policy=None):
    """Mean across supported (pattern,condition); each domain has exactly half mass."""
    policy = policy or SupportPolicy()
    qs, qt = (source_members.detach(), target_members.detach())
    if source_logits.shape != qs.shape or target_logits.shape != qt.shape or qs.shape[1] != qt.shape[1]:
        raise ValueError('Conditional logits/membership dimensions differ')
    if not all((torch.isfinite(q).all() and (not (q < 0).any()) for q in (qs, qt))):
        raise ValueError('Membership must be finite and nonnegative')
    ps, pt = (pattern_ids(source_availability), pattern_ids(target_availability))
    losses, regions = ([], [])
    loss_s = F.binary_cross_entropy_with_logits(source_logits, torch.ones_like(source_logits), reduction='none')
    loss_t = F.binary_cross_entropy_with_logits(target_logits, torch.zeros_like(target_logits), reduction='none')
    for pattern in sorted(set(ps.tolist()) | set(pt.tolist())):
        for k in range(qs.shape[1]):
            sw = qs[:, k] * (ps == pattern)
            tw = qt[:, k] * (pt == pattern)
            sok, ss = policy.check(sw)
            tok, ts = policy.check(tw)
            valid = bool(pattern and sok and tok)
            region = dict(pattern=pattern, condition=k, source=ss, target=ts, supported=valid, reason='common_support' if valid else 'no_available_source' if not pattern else 'insufficient_joint_support')
            if valid:
                value = 0.5 * ((loss_s[:, k] * sw).sum() / sw.sum() + (loss_t[:, k] * tw).sum() / tw.sum())
                losses.append(value)
                region['loss'] = float(value.detach())
            regions.append(region)
    zero = (source_logits.sum() + target_logits.sum()) * 0
    loss = torch.stack(losses).mean() if losses else zero
    diagnostics = dict(supported_regions=len(losses), total_regions=len(regions), regions=regions, source_patterns={str(p): int((ps == p).sum()) for p in ps.unique().tolist()}, target_patterns={str(p): int((pt == p).sum()) for p in pt.unique().tolist()}, reduction='mean of supported pattern/condition pairs; each domain normalized separately', support_policy=asdict(policy))
    return (loss, diagnostics)

class ConditionalAdapter(nn.Module):

    def __init__(self, dim, conditions, support=None):
        super().__init__()
        if conditions < 1:
            raise ValueError('Positive condition count required')
        self.support = support or SupportPolicy()
        self.global_critic = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.local_stem = PathStem(dim)
        self.local_heads = nn.Linear(dim, conditions)

    def forward(self, source, target, source_members, target_members, *, lambda_adv, omega_local):
        if not math.isfinite(lambda_adv) or not 0 <= lambda_adv <= 1:
            raise ValueError('lambda_adv outside [0,1]')
        if not math.isfinite(omega_local) or not 0 <= omega_local <= 1:
            raise ValueError('omega_local outside [0,1]')
        global_logits, local_logits = ([], [])
        for out in (source, target):
            z = ReverseLayerF.apply(out['z_adv'], lambda_adv)
            global_logits.append(self.global_critic(z))
            paths = ReverseLayerF.apply(out['paths'], lambda_adv)
            local = self.local_stem(paths, out['availability'])
            local_logits.append(self.local_heads(local))
        avail = (source['availability'], target['availability'])
        ones = [torch.ones_like(v) for v in global_logits]
        global_loss, global_info = conditional_domain_loss(*global_logits, *ones, *avail, self.support)
        local_loss, local_info = conditional_domain_loss(*local_logits, source_members, target_members, *avail, self.support)
        omega_used = omega_local if local_info['supported_regions'] else 0.0
        if not global_info['supported_regions']:
            omega_used = 0.0
            local_loss = local_loss * 0
        loss = (1 - omega_used) * global_loss + omega_used * local_loss
        return (loss, dict(global_loss=float(global_loss.detach()), local_loss=float(local_loss.detach()), global_support=global_info, local_support=local_info, omega_local=omega_used, requested_omega_local=omega_local, lambda_adv=lambda_adv, lambda_adv_times_omega_local=lambda_adv * omega_used, skipped_adaptation=not bool(global_info['supported_regions'])))

@dataclass
class OmegaScheduler:
    omega_local: float = 0.5
    ema_decay: float = 0.9
    epsilon: float = 1e-08

    def __post_init__(self):
        if not all((math.isfinite(v) for v in (self.omega_local, self.ema_decay, self.epsilon))) or not 0 <= self.omega_local <= 1 or (not 0 <= self.ema_decay < 1) or (self.epsilon <= 0):
            raise ValueError('Invalid omega scheduler')

    def update(self, global_distance, local_distances, *, local_available=True):
        previous = self.omega_local
        if not local_available or not local_distances:
            return dict(omega_local=0.0, ema_state=previous, raw_omega=None, reason='global_only_no_local', global_distance=global_distance, local_distances=list(local_distances))
        values = [global_distance, *local_distances]
        if any((v is None or not math.isfinite(v) or (not 0 <= v <= 2) for v in values)):
            raise ValueError('Only finite scheduling PAD in [0,2], never BCE/raw negative PAD')
        local = sum(local_distances) / len(local_distances)
        if global_distance + local <= self.epsilon:
            raw, reason = (None, 'both_near_zero_keep_previous')
        else:
            raw = local / (global_distance + local + self.epsilon)
            self.omega_local = self.ema_decay * previous + (1 - self.ema_decay) * raw
            reason = 'updated'
        assert 0 <= self.omega_local <= 1
        return dict(omega_local=self.omega_local, ema_state=self.omega_local, previous=previous, raw_omega=raw, global_distance=global_distance, local_distance_mean=local, local_distances=list(local_distances), reason=reason)
